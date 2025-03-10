{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    numpy
    pandas
    scikit-learn
    uvicorn
    fastapi
    python-multipart
  ]);
in

pkgs.mkShell {
  buildInputs = [
    pythonEnv
  ];
}
