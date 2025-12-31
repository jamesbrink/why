{
  description = "why - quick error explanation CLI using local LLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, flake-utils, crane }:
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
        # Crane setup for incremental Rust builds
        # ============================================================
        craneLib = crane.mkLib pkgs;

        # Source filtering - include Rust files plus prompt templates
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            # Include prompt templates (required by include_str!)
            (builtins.match ".*\.txt$" path != null) ||
            # Include standard Rust/Cargo files
            (craneLib.filterCargoSources path type);
        };

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
            family = "qwen";
          };

          # Qwen3 0.6B - newer model, similar size
          qwen3-0_6b = {
            name = "Qwen3-0.6B-Q8_0.gguf";
            url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";
            sha256 = "0cdh7c26vlcv4l3ljrh7809cfhvh2689xfdlkd6kbmdd48xfcrcl";
            description = "Qwen3 0.6B (Q8_0) - ~639MB, newest Qwen model";
            family = "qwen";
          };

          # SmolLM2 135M - tiny model for fast experimentation
          smollm2-135m = {
            name = "SmolLM2-135M-Instruct-Q8_0.gguf";
            url = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf";
            sha256 = "10xsdfq2wx0685kd7xx9hw4xha0jkcdmi60xqlf784vrdxqra4ss";
            description = "SmolLM2 135M Instruct (Q8_0) - ~145MB, fastest/smallest";
            family = "smollm";
          };

          # Gemma 3 270M - Google's small model
          gemma3-270m = {
            name = "gemma-3-270m-it-Q8_0.gguf";
            url = "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf";
            sha256 = "164nkcwi7b8aca9a45qgs2w8mwhz1z11qz1wsnqw2y9gkwasamni";
            description = "Gemma 3 270M Instruct (Q8_0) - ~292MB, Google";
            family = "gemma";
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
        # Common build arguments for Crane
        # ============================================================
        commonArgs = {
          inherit src;
          strictDeps = true;

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

          cargoExtraArgs = "--features ${gpuFeaturesStr}";
        };

        # ============================================================
        # Build dependencies only (cached separately from source)
        # This is the key to Crane's speed - deps are rebuilt only
        # when Cargo.lock changes, not when source changes
        # ============================================================
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # ============================================================
        # Build the why CLI (no embedded model)
        # ============================================================
        why-cli = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;

          # Don't run tests during build (we have a separate check)
          doCheck = false;

          # Clean llama-cpp-sys-2 build artifacts to avoid path issues
          # CMake generates files with hardcoded absolute paths that reference
          # the deps-only build directory, which doesn't exist in the main build
          preBuild = ''
            rm -rf target/release/build/llama-cpp-sys-2-* 2>/dev/null || true
            rm -rf target/debug/build/llama-cpp-sys-2-* 2>/dev/null || true
          '';

          meta = with pkgs.lib; {
            description = "Quick error explanation CLI using local LLM";
            homepage = "https://github.com/jamesbrink/why";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.unix;
            mainProgram = "why";
          };
        });

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
              echo "Family: ${model.family}"

              cat ${why-cli}/bin/why > why-embedded

              cat ${modelFile} >> why-embedded

              # Trailer format: WHYMODEL (8) + offset (8) + size (8) + family (1) = 25 bytes
              # Family: 0=qwen, 1=gemma, 2=smollm
              python3 -c "
              import struct
              import sys
              offset = $BINARY_SIZE
              size = $MODEL_SIZE
              family_map = {'qwen': 0, 'gemma': 1, 'smollm': 2}
              family = family_map.get('${model.family}', 0)
              trailer = b'WHYMODEL' + struct.pack('<Q', offset) + struct.pack('<Q', size) + struct.pack('<B', family)
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
            echo "  gemma3-270m         - ${models.gemma3-270m.description}"
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
            gemma3-270m|gemma3|gemma)
              URL="${models.gemma3-270m.url}"
              NAME="${models.gemma3-270m.name}"
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

          # Gemma 3 variant - Google's small model
          why-gemma3 = mkEmbeddedWhy {
            model = models.gemma3-270m;
            pname = "why-gemma3";
          };
        };

        # ============================================================
        # Checks - run tests, clippy, formatting
        # Note: why-cli removed to reduce parallel builds (OOM on CI)
        # ============================================================
        checks = {
          # Run clippy (also verifies compilation)
          why-clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            # Clean llama-cpp-sys-2 build artifacts
            preBuild = ''
              rm -rf target/release/build/llama-cpp-sys-2-* 2>/dev/null || true
              rm -rf target/debug/build/llama-cpp-sys-2-* 2>/dev/null || true
            '';
          });

          # Check formatting
          why-fmt = craneLib.cargoFmt {
            inherit src;
          };

          # Run tests
          why-test = craneLib.cargoTest (commonArgs // {
            inherit cargoArtifacts;
            # Clean llama-cpp-sys-2 build artifacts
            preBuild = ''
              rm -rf target/release/build/llama-cpp-sys-2-* 2>/dev/null || true
              rm -rf target/debug/build/llama-cpp-sys-2-* 2>/dev/null || true
            '';
          });
        };

        # Development shell
        devShells.default = craneLib.devShell {
          # Inherit checks to get build inputs
          checks = self.checks.${system};

          # Additional dev tools
          packages = with pkgs; [
            # LLM inference
            llama-cpp

            # Rust toolchain (crane provides rustc, cargo, etc.)
            rustfmt
            clippy
            rust-analyzer
            cargo-tarpaulin

            # Used by embed.sh for size calculations
            bc

            # Python tools
            ruff
            python3
            python3Packages.pyyaml

            # Helper scripts
            buildScript
            fetchModelScript
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          shellHook = ''
            echo "why development shell (powered by Crane)"
            echo "GPU support: ${gpuFeaturesStr}"
            echo ""
            echo "Commands:"
            echo "  build              - Build and embed default model"
            echo "  fetch-model <name> - Download a model for experimentation"
            echo "  cargo build --features ${gpuFeaturesStr}"
            echo ""
            echo "Crane benefits:"
            echo "  - Dependencies cached separately from source"
            echo "  - Incremental rebuilds when only source changes"
            echo "  - nix build is much faster on subsequent runs"
            echo ""
            echo "Quick experimentation:"
            echo "  1. fetch-model qwen3-0.6b"
            echo "  2. cargo run --features ${gpuFeaturesStr} -- --model Qwen3-0.6B-Q8_0.gguf 'your error'"
          '';
        };
      }
    );
}
