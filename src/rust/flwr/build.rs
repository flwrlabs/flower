fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_src = format!("{}/../../proto/flwr/proto/fleet.proto", manifest_dir);
    let proto_include = format!("{}/../../proto", manifest_dir);

    tonic_build::configure()
        .build_server(false)
        .out_dir(format!("{}/src", manifest_dir))
        .compile(&[&proto_src], &[&proto_include])
        .unwrap();
}
