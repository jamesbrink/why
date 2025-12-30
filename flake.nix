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

        # Read version from Cargo.toml
        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        version = cargoToml.package.version;

        # Git revision for version string (works with dirty trees too)
        gitRev = self.shortRev or self.dirtyShortRev or "unknown";

        # ============================================================
        # Model Definitions
        # ============================================================
        # Each model has: name, url, sha256, and optional description
        models = {
          # Default model - Qwen2.5-Coder 0.5B (Q8_0 quantization)
          qwen2_5-coder-0_5b = {
            name = "qwen2.5-coder-0.5b-instruct-q8_0.gguf";
            url = "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q8_0.gguf";
            sha256 = "1la4ndkiywa6swigj60y4xpsxd0zr3p270l747qi5m4pz8hpg9z1";
            description = "Qwen2.5-Coder 0.5B Instruct (Q8_0) - ~530MB, good quality";
          };

          # Qwen3 0.6B - newer model, similar size
          qwen3-0_6b = {
            name = "Qwen3-0.6B-Q8_0.gguf";
            url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";
            sha256 = "0cdh7c26vlcv4l3ljrh7809cfhvh2689xfdlkd6kbmdd48xfcrcl";
            description = "Qwen3 0.6B (Q8_0) - ~639MB, newest Qwen model";
          };

          # SmolLM2 135M - tiny model for fast experimentation
          smollm2-135m = {
            name = "SmolLM2-135M-Instruct-Q8_0.gguf";
            url = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf";
            sha256 = "10xsdfq2wx0685kd7xx9hw4xha0jkcdmi60xqlf784vrdxqra4ss";
            description = "SmolLM2 135M Instruct (Q8_0) - ~145MB, fastest/smallest";
          };
        };

        # Default model selection
        defaultModel = models.qwen2_5-coder-0_5b;

        # GPU features based on platform
        gpuFeatures = if isDarwin then [ "metal" ] else [ "vulkan" ];
        gpuFeaturesStr = builtins.concatStringsSep "," gpuFeatures;

        # Platform-specific native build inputs for llama-cpp-sys-2
        darwinBuildInputs = [
          pkgs.apple-sdk_15
          pkgs.darwin.cctools
        ];

        linuxBuildInputs = with pkgs; [
          vulkan-headers
          vulkan-loader
          shaderc
          glslang
        ];

        # ============================================================
        # Build the why CLI (no embedded model)
        # ============================================================
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
            shaderc
          ] else []);

          buildInputs = with pkgs; [
            openssl
          ] ++ (if isDarwin then darwinBuildInputs else linuxBuildInputs);

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          WHY_GIT_SHA = gitRev;

          buildFeatures = gpuFeatures;

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

        # ============================================================
        # Function to create embedded binary with a specific model
        # ============================================================
        mkEmbeddedWhy = { model, pname ? "why-embedded" }:
          let
            modelFile = pkgs.fetchurl {
              url = model.url;
              sha256 = model.sha256;
            };
          in
          pkgs.stdenv.mkDerivation {
            inherit pname version;

            dontUnpack = true;
            dontStrip = true;

            nativeBuildInputs = [ pkgs.python3 pkgs.installShellFiles ];

            buildPhase = ''
              runHook preBuild

              BINARY_SIZE=$(stat -c%s ${why-cli}/bin/why 2>/dev/null || stat -f%z ${why-cli}/bin/why)
              MODEL_SIZE=$(stat -c%s ${modelFile} 2>/dev/null || stat -f%z ${modelFile})

              echo "Binary: ${why-cli}/bin/why ($BINARY_SIZE bytes)"
              echo "Model: ${model.name} ($MODEL_SIZE bytes)"

              cat ${why-cli}/bin/why > why-embedded

              cat ${modelFile} >> why-embedded

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

              installShellCompletion --cmd why \
                --bash <($out/bin/why --completions bash) \
                --zsh <($out/bin/why --completions zsh) \
                --fish <($out/bin/why --completions fish)
            '';

            meta = why-cli.meta // {
              description = "Quick error explanation CLI with embedded LLM model (${model.name})";
              mainProgram = "why";
            };
          };

        # ============================================================
        # Build script for development
        # ============================================================
        buildScript = pkgs.writeShellScriptBin "build" ''
          set -euo pipefail

          MODEL="''${1:-${defaultModel.name}}"
          MODEL_URL="${defaultModel.url}"

          if [[ ! -f "$MODEL" ]]; then
            echo "Model not found, downloading from HuggingFace..."
            ${pkgs.curl}/bin/curl -L -o "$MODEL" "$MODEL_URL"
            echo ""
          fi

          echo "Building debug binary with GPU support: ${gpuFeaturesStr}"
          cargo build --features ${gpuFeaturesStr}
          echo ""
          echo "Building release binary with GPU support: ${gpuFeaturesStr}"
          cargo build --release --features ${gpuFeaturesStr}
          echo ""
          echo "Embedding model..."
          ./scripts/embed.sh target/release/why "$MODEL" why-embedded
        '';

        # Script to download models for experimentation
        fetchModelScript = pkgs.writeShellScriptBin "fetch-model" ''
          set -euo pipefail

          usage() {
            echo "Usage: fetch-model <model-name>"
            echo ""
            echo "Available models:"
            echo "  qwen2.5-coder-0.5b  - ${models.qwen2_5-coder-0_5b.description}"
            echo "  qwen3-0.6b          - ${models.qwen3-0_6b.description}"
            echo "  smollm2-135m        - ${models.smollm2-135m.description}"
            echo ""
            echo "Or provide a direct URL to a GGUF file"
          }

          if [[ $# -lt 1 ]]; then
            usage
            exit 1
          fi

          case "$1" in
            qwen2.5-coder-0.5b|qwen2.5)
              URL="${models.qwen2_5-coder-0_5b.url}"
              NAME="${models.qwen2_5-coder-0_5b.name}"
              ;;
            qwen3-0.6b|qwen3)
              URL="${models.qwen3-0_6b.url}"
              NAME="${models.qwen3-0_6b.name}"
              ;;
            smollm2-135m|smollm2|smol)
              URL="${models.smollm2-135m.url}"
              NAME="${models.smollm2-135m.name}"
              ;;
            http*://*.gguf)
              URL="$1"
              NAME="$(basename "$URL")"
              ;;
            *)
              echo "Unknown model: $1"
              usage
              exit 1
              ;;
          esac

          if [[ -f "$NAME" ]]; then
            echo "Model already exists: $NAME"
          else
            echo "Downloading $NAME..."
            ${pkgs.curl}/bin/curl -L -o "$NAME" "$URL"
            echo "Downloaded: $NAME"
          fi
        '';

      in
      {
        packages = {
          # Default: embedded with qwen2.5-coder
          default = mkEmbeddedWhy { model = defaultModel; };

          # Bare CLI without model (for experimentation with --model flag)
          cli = why-cli;

          # Alias for default
          why = mkEmbeddedWhy { model = defaultModel; };

          # Explicit variant names (only include models with valid hashes)
          why-qwen2_5-coder = mkEmbeddedWhy {
            model = models.qwen2_5-coder-0_5b;
            pname = "why-qwen2_5-coder";
          };

          # Qwen3 variant
          why-qwen3 = mkEmbeddedWhy {
            model = models.qwen3-0_6b;
            pname = "why-qwen3";
          };

          # SmolLM2 variant - tiny and fast
          why-smollm2 = mkEmbeddedWhy {
            model = models.smollm2-135m;
            pname = "why-smollm2";
          };
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

            # bindgen
            rustPlatform.bindgenHook

            # Used by embed.sh for size calculations
            bc

            # Python tools
            ruff
            python3
            python3Packages.pyyaml

            # Helper scripts
            buildScript
            fetchModelScript
          ] ++ (if isDarwin then [
            apple-sdk_15
            darwin.cctools
          ] else [
            vulkan-headers
            vulkan-loader
            shaderc
            glslang
          ]);

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            echo "why development shell"
            echo "GPU support: ${gpuFeaturesStr}"
            echo ""
            echo "Commands:"
            echo "  build              - Build and embed default model"
            echo "  fetch-model <name> - Download a model for experimentation"
            echo "  cargo build --features ${gpuFeaturesStr}"
            echo ""
            echo "Quick experimentation:"
            echo "  1. fetch-model qwen3-0.6b"
            echo "  2. cargo run --features ${gpuFeaturesStr} -- --model Qwen3-0.6B-Q8_0.gguf 'your error'"
          '';
        };
      }
    );
}
