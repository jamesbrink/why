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

        # Qwen2.5-Coder GGUF model (tracked via git-lfs)
        qwen-model = ./qwen2.5-coder-0.5b.gguf;

        # Platform-specific native build inputs for llama-cpp-sys-2
        # New Darwin SDK pattern: just add apple-sdk and it provides all frameworks
        darwinBuildInputs = [
          pkgs.apple-sdk_15
          pkgs.darwin.cctools
        ];

        linuxBuildInputs = with pkgs; [
          # For potential CUDA/OpenCL support on Linux
        ];

        # Build the why CLI
        why-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "why";
          version = "0.1.0";

          src = pkgs.lib.cleanSource ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            rustPlatform.bindgenHook
          ];

          buildInputs = with pkgs; [
            openssl
          ] ++ (if isDarwin then darwinBuildInputs else linuxBuildInputs);

          # Environment variables for the build
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

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
          version = "0.1.0";

          # No source needed - we're just combining artifacts
          dontUnpack = true;

          # Don't strip - it removes the embedded model!
          dontStrip = true;

          nativeBuildInputs = [ pkgs.python3 ];

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
          '';

          meta = why-cli.meta // {
            description = "Quick error explanation CLI with embedded LLM model";
            mainProgram = "why";
          };
        };

        # Helper script to run full build (debug + release with embedded model)
        buildScript = pkgs.writeShellScriptBin "build" ''
          set -euo pipefail
          echo "Building debug binary..."
          cargo build
          echo ""
          echo "Building release binary..."
          cargo build --release
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

            # Helper scripts
            buildScript
          ] ++ (if isDarwin then [
            # macOS specific - new SDK provides frameworks and libiconv
            apple-sdk_15
            darwin.cctools
          ] else []);

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };
      }
    );
}
