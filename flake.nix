{
  description = "Development environment for QCH-VulDet";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv

          # Native build tools used when a wheel is unavailable.
          gcc
          ninja
          pkg-config
        ] ++ runtimeLibs;

        shellHook = ''
          export UV_PYTHON="${pkgs.python312}/bin/python3.12"
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:''${LD_LIBRARY_PATH:-}"

          echo "Python: $(python --version)"
          echo "uv uses: $UV_PYTHON"
        '';
      };
    };
}
