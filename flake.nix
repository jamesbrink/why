{
  description = "why - quick error explanation CLI using local LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    {
      # Overlay for easy integration into NixOS/home-manager configs
      overlays.default = final: prev: {
        why = self.packages.${final.system}.default;
      };
    } //
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        isDarwin = pkgs.stdenv.isDarwin;
        isLinux = pkgs.stdenv.isLinux;
        isAarch64 = pkgs.stdenv.hostPlatform.isAarch64;

        # Read version from Cargo.toml
        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        version = cargoToml.package.version;

        # Qwen2.5-Coder GGUF model (fetched directly to avoid git-lfs pointer files)
        qwen-model = pkgs.fetchurl {
          url = "https://media.githubusercontent.com/media/jamesbrink/why/refs/heads/main/qwen2.5-coder-0.5b.gguf";
          sha256 = "1q5dgipixb13qp54hsbwm00cgqc9mfv56957cgi08cy60bmkls90";
        };

        # GPU features based on platform
        # Linux: Vulkan (AMD/Intel/NVIDIA via Vulkan drivers)
        # macOS: Metal (Apple GPU)
        gpuFeatures = if isDarwin then [ "metal" ] else [ "vulkan" ];
        gpuFeaturesStr = builtins.concatStringsSep "," gpuFeatures;

        # Platform-specific native build inputs for llama-cpp-sys-2
        # New Darwin SDK pattern: just add apple-sdk and it provides all frameworks
        darwinBuildInputs = [
          pkgs.apple-sdk_15
          pkgs.darwin.cctools
        ];

        linuxBuildInputs = with pkgs; [
          # Vulkan support (AMD/Intel/NVIDIA)
          vulkan-headers
          vulkan-loader
          shaderc
          glslang
        ];

        # Build the why CLI
        why-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "why";
          inherit version;

          src = pkgs.lib.cleanSource ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            rustPlatform.bindgenHook
          ] ++ (if isLinux then [
            # Vulkan shader compiler (glslc) - must be in nativeBuildInputs for CMake to find it
            shaderc
          ] else []);

          buildInputs = with pkgs; [
            openssl
          ] ++ (if isDarwin then darwinBuildInputs else linuxBuildInputs);

          # Environment variables for the build
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # Enable GPU acceleration via cargo features
          buildFeatures = gpuFeatures;

          # Disable running tests during build (they need the model)
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Quick error explanation CLI using local LLM";
            homepage = "https://github.com/jamesbrink/why";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.unix;
            mainProgram = "why";
          };
        };

        # Create the embedded binary with model baked in
        why-embedded = pkgs.stdenv.mkDerivation {
          pname = "why-embedded";
          inherit version;

          # No source needed - we're just combining artifacts
          dontUnpack = true;

          # Don't strip - it removes the embedded model!
          dontStrip = true;

          nativeBuildInputs = [ pkgs.python3 pkgs.installShellFiles ];

          buildPhase = ''
            runHook preBuild

            # Get file sizes
            BINARY_SIZE=$(stat -c%s ${why-cli}/bin/why 2>/dev/null || stat -f%z ${why-cli}/bin/why)
            MODEL_SIZE=$(stat -c%s ${qwen-model} 2>/dev/null || stat -f%z ${qwen-model})

            echo "Binary: ${why-cli}/bin/why ($BINARY_SIZE bytes)"
            echo "Model: ${qwen-model} ($MODEL_SIZE bytes)"

            # Copy binary (use cat to create writable copy)
            cat ${why-cli}/bin/why > why-embedded

            # Append model
            cat ${qwen-model} >> why-embedded

            # Write trailer using Python for reliable little-endian encoding
            python3 -c "
            import struct
            import sys
            offset = $BINARY_SIZE
            size = $MODEL_SIZE
            trailer = b'WHYMODEL' + struct.pack('<Q', offset) + struct.pack('<Q', size)
            sys.stdout.buffer.write(trailer)
            " >> why-embedded

            chmod +x why-embedded

            FINAL_SIZE=$(stat -c%s why-embedded 2>/dev/null || stat -f%z why-embedded)
            echo "Done! Final size: $FINAL_SIZE bytes"

            runHook postBuild
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp why-embedded $out/bin/why

            # Generate and install shell completions
            installShellCompletion --cmd why \
              --bash <($out/bin/why --completions bash) \
              --zsh <($out/bin/why --completions zsh) \
              --fish <($out/bin/why --completions fish)
          '';

          meta = why-cli.meta // {
            description = "Quick error explanation CLI with embedded LLM model";
            mainProgram = "why";
          };
        };

        # Helper script to run full build (debug + release with embedded model)
        buildScript = pkgs.writeShellScriptBin "build" ''
          set -euo pipefail
          echo "Building debug binary with GPU support: ${gpuFeaturesStr}"
          cargo build --features ${gpuFeaturesStr}
          echo ""
          echo "Building release binary with GPU support: ${gpuFeaturesStr}"
          cargo build --release --features ${gpuFeaturesStr}
          echo ""
          echo "Embedding model..."
          ./scripts/embed.sh target/release/why qwen2.5-coder-0.5b.gguf why-embedded
        '';

      in
      {
        # Default package is the embedded version
        packages = {
          default = why-embedded;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          name = "why-dev";

          nativeBuildInputs = with pkgs; [
            # LLM inference
            llama-cpp

            # Rust toolchain
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer
            cargo-tarpaulin

            # Native deps for llama.cpp Rust bindings
            pkg-config
            openssl
            cmake

            # bindgen needs this to find C headers (critical for llama-cpp-sys-2)
            rustPlatform.bindgenHook

            # Used by embed.sh for size calculations
            bc

            # Helper scripts
            buildScript
          ] ++ (if isDarwin then [
            # macOS specific - new SDK provides frameworks and libiconv
            apple-sdk_15
            darwin.cctools
          ] else [
            # Linux GPU support
            vulkan-headers
            vulkan-loader
            shaderc
            glslang
          ]);

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            echo "GPU support: ${gpuFeaturesStr}"
            echo "Use 'build' script or 'cargo build --features ${gpuFeaturesStr}'"
          '';
        };
      }
    );
}
