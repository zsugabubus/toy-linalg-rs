use clap::{App, value_t, SubCommand, Arg};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;

mod linalg;
use self::linalg::*;

fn main() {

    let matches = App::new("linalg")
        .arg(Arg::with_name("file")
             .short("f")
             .long("file")
             .help("Sets file to read input elements from")
             .long_help(
                 "Sets the input file. Note: Use \"-\" for stdin.")
             .value_name("FILE")
             .required(true))
        .arg(Arg::with_name("print")
             .short("p")
             .long("print")
             .help("Print input matrixes"))

        .subcommand(SubCommand::with_name("iterative")
                    .visible_alias("iter")
                    .about("Use iterative solvers")
                    .arg(Arg::with_name("algorithm")
                         .short("a")
                         .long("algorithm")
                         .value_name("NAME")
                         .possible_values(&["jacobi", "gauss-seidel"])
                         .required(true)
                         .help("Iterative algorithm to use"))
                    .arg(Arg::with_name("stop")
                         .short("s")
                         .long("stop")
                         .value_name("COUNT")
                         .validator(|val| match val.parse::<usize>() {
                                Ok(_) => Ok(()),
                                _ => Err(String::from("must be a non-negative ingeter")),
                            })
                         .help("Stop iteration after `COUNT` steps if diverges"))
                    .arg(Arg::with_name("omega")
                         .short("o")
                         .long("omega")
                         .value_name("VALUE")
                         .validator(|val| match val.parse::<f64>() {
                                Ok(o) if o > 0.0 => Ok(()),
                                _ => Err(String::from("must be a positive number")),
                            })
                          .help("Sets parameter omega of algorithm"))
                    .arg(Arg::with_name("iter-count")
                         .short("i")
                         .long("iters")
                         .value_name("COUNT")
                         .validator(|val| match val.parse::<usize>() {
                                Ok(_) => Ok(()),
                                _ => Err(String::from("must be a non-negative ingeter")),
                            })
                         .help("Sets iteration count")))
        .get_matches();

    let (a, mut x, b) = if let Some(file) = matches.value_of("file") {
        read_combo(file).unwrap()
    } else {
        //use rand::{thread_rng, Rng};
        // for i in 0..10000 {
        //     let num: f64 = thread_rng().gen_range(-7e4, 1.3e5);
        //     println!("{}", i);
        //     a.add((i, i), num);
        // }

        // for x in 0..10000 {
        //     let col: Index = thread_rng().gen_range(0, 10000);
        //     let row: Index = thread_rng().gen_range(0, 10000);
        //     if col == row { continue; }
        //     let num: f64 = thread_rng().gen_range(-7e4, 1.3e5);
        //     a.add((col, row), num);
        // }
        unimplemented!();
    };

    if matches.is_present("print") {
        println!("A: {}", a);
        println!("x0: {}", x);
        println!("b: {}", b);
    }

    if let Some(matches) = matches.subcommand_matches("iterative") {

        let iter_count = value_t!(matches, "iter-count", usize).unwrap_or(100);
        // TODO: find a better name
        let div_limit = value_t!(matches, "stop", usize).unwrap_or(10);
        let omega = value_t!(matches, "omega", f64).unwrap_or(1.0);

        let result = IterativeMethod::build(&a, &mut x, &b)
            .omega(omega)
            .method({
                match matches.value_of("algorithm") {
                    Some("gauss-seidel") => Method::GaussSeidel,
                    Some("jacobi") => Method::Jacobi,
                    _ => unimplemented!(),
                }
            })
            .unwrap().expect("incompatible matrix shapes")
            .take(iter_count)
            .try_fold((1, None, None), |k, q| {
                let now = Instant::now();

                eprintln!("step {:4}: q={:.32?} \u{0394}={:#?}", k.0, q.unwrap_or(std::f64::NAN), now - k.1.unwrap_or(now));
                match (q, div_limit) {
                    (Some(q), s) if k.0 > s && q >= 1.0 => Err(q),
                    (Some(q), _) if q == 0.0 => Err(q),
                    _ => Ok((k.0 + 1, Some(now), q)),
                }
            });

        let q = match result {
            Err(q) => Some(q),
            Ok((_, _, q)) => q,
        };

        match q {
            Some(q) if q >= 1.0 => eprint!("warn: algorithm diverges\n"),
            _ => {},
        }

        println!("x: {}", x);
    }

}

fn read_vector<Iter: Iterator<Item=String>>(iter: &mut Iter, v: &mut Vector) {
    (0..v.shape()).for_each(|i| {
        v[i] = iter.next().unwrap().parse().expect("scalar value")
    });
}

fn read_combo(path: &str) -> Result<(SparseMatrix, Vector, Vector), ()> {
    let file = match path {
        "-" => Box::new(io::stdin()) as Box<dyn io::Read>,
        _ => {
            let file = File::open(path).expect("open file");
            Box::new(file) as Box<dyn io::Read>
        }
    };
    let mut lines = BufReader::new(file).lines().scan((), |_, line| line.ok());
    let size: Index = lines.next().unwrap().parse().expect("matrix size");

    let mut a = SparseMatrix::new((size, size));

    let m: Index = lines.next().unwrap().parse().expect("sparse matrix element count");
    for _ in 0..m {
        let line = lines.next().unwrap();
        let mut tokens = line.split_whitespace();
        let row: Index = tokens.next().and_then(|t| t.parse().ok()).expect("row index");
        let col: Index = tokens.next().and_then(|t| t.parse().ok()).expect("column index");
        let value: Scalar = tokens.next().and_then(|t| t.parse().ok()).expect("scalar value for `a' matrix");

        a.add((col - 1, row - 1), value);
    }

    let b = unsafe {
        let mut v = Vector::new_uninitialized(size);
        read_vector(&mut lines, &mut v);
        v
    };

    let x = unsafe {
        let mut v = Vector::new_uninitialized(size);
        read_vector(&mut lines, &mut v);
        v
    };


    Ok((a, x, b))
}
