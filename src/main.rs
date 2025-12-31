use clap::Parser;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .sg file to run
    file: String,
}

fn main() {
    let args = Args::parse();
    
    // If file doesn't exist, treat as source
    let source = if std::path::Path::new(&args.file).exists() {
        std::fs::read_to_string(&args.file).expect("Failed to read file")
    } else {
        args.file.clone()
    };

    println!("{}", sigma::run_file(&source));
}
