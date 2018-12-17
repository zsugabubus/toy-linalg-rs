use clap::{App, Arg, SubCommand, crate_name, crate_version, crate_authors, crate_description};

pub fn build_cli() -> App<'static, 'static> {

    fn is_valid_integer(value: String) -> Result<(), String> {
        match value.parse::<usize>() {
            Ok(o) => Ok(()),
            _ => Err(String::from("must be a non-negative integer")),
        }
    }

    fn is_valid_omega(value: String) -> Result<(), String> {
        match value.parse::<f64>() {
            Ok(o) if o > 0.0 => Ok(()),
            _ => Err(String::from("must be a positive number")),
        }
    }

    let iterative_solver_subcmd = SubCommand::with_name("iterative")
        .visible_alias("iter")
            .about("Use iterative solvers")
            .arg(Arg::with_name("algorithm")
                 .display_order(2)
                 .short("a")
                 .long("algorithm")
                 .value_name("NAME")
                 .possible_values(&["jacobi", "gauss-seidel"])
                 .required(true)
                 .help("Sets the iterative algorithm to use"))
            .arg(Arg::with_name("omega")
                 .display_order(3)
                 .short("o")
                 .long("omega")
                 .value_name("VALUE")
                 .validator(is_valid_omega)
                 .help("Sets parameter omega of algorithm"))
            .arg(Arg::with_name("iters")
                 .display_order(4)
                 .short("i")
                 .long("iters")
                 .value_name("COUNT")
                 .validator(is_valid_integer)
                 .help("Sets iteration count"))
            .arg(Arg::with_name("stop")
                 .display_order(5)
                 .short("s")
                 .long("stop")
                 .value_name("COUNT")
                 .validator(is_valid_integer)
                 .help("Stop iteration after `COUNT` steps if diverges"));

    let test_subcmd = SubCommand::with_name("test")
        .about("Test")
        .arg(Arg::with_name("algorithm")
             .display_order(1)
             .short("a")
             .long("algorithm")
             .value_name("NAME")
             .possible_values(&["jacobi", "gauss-seidel"])
             .required(true)
             .help("Sets the algorithm to use"))
        .arg(Arg::with_name("iters")
             .display_order(2)
             .short("i")
             .long("iters")
             .value_name("COUNT")
             .validator(is_valid_integer)
             .required(true)
             .help("Sets iteration count"))
        .arg(Arg::with_name("size")
             .display_order(3)
             .short("s")
             .long("size")
             .value_names(&["FROM", "TO"])
             .required(true)
             .help("Sets test range to [from, to)"))
        .arg(Arg::with_name("range")
             .display_order(4)
             .short("r")
             .long("range")
             .value_names(&["MIN", "MAX"])
             .required(true)
             .help("Sets the value range of elements in vector x"))
        .arg(Arg::with_name("omega")
             .short("o")
             .long("omega")
             .value_name("VALUE")
             .validator(is_valid_omega)
             .help("Sets parameter omega of the algorithm"));

    let solve_subcmd = SubCommand::with_name("solve")
        .about("Solve Ax=b equation for x")
        .arg(Arg::with_name("file")
             .display_order(1)
             .short("f")
             .long("file")
             .value_name("FILE")
             .required(true)
             .help("Sets file to read inputs from")
             .long_help(
                 "Sets the input file. Note: Use \"-\" for stdin."
             ))
        .arg(Arg::with_name("print")
                 .long("print")
                 .requires("file")
                 .help("Prints input matrixes"))
        .subcommand(iterative_solver_subcmd);

    App::new(crate_name!())
        .version(crate_version!())
        .author(crate_authors!("\n"))
        .about(crate_description!())
        .subcommand(solve_subcmd)
        .subcommand(test_subcmd)
}


