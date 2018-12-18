use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;

mod linalg;
use self::linalg::*;
mod cli;

fn main() {
    use clap::{value_t, values_t};

    let matches = cli::build_cli().get_matches();

    match matches.subcommand() {
        ("decompose", Some(decomp_matches)) => {
            use self::linalg::decomp::*;
            
            let a = if let Some(file) = decomp_matches.value_of("file") {
                read_square_matrix(file).unwrap()
            } else {
                unimplemented!();
            };

            if decomp_matches.is_present("print") {
                println!("A: {}", a);
            }

            match decomp_matches.value_of("algorithm") {
                Some("qr-gramschmidt") => {
                    let (q, r) = QR::decompose(&a, QRMethod::GramSchmidt).into_parts();
                    println!("Q: {}", q);
                    println!("R: {}", r);
                },
                Some("qr-housholder") => {
                    let (q, r) = QR::decompose(&a, QRMethod::Householder).into_parts();
                    println!("Q: {}", q);
                    println!("R: {}", r);
                },
                _ => unimplemented!(),
            }
        },
        ("solve", Some(solve_matches)) => {

            let (a, mut x, b) = if let Some(file) = solve_matches.value_of("file") {
                read_combo(file).unwrap()
            } else {
                unimplemented!();
            };

            if solve_matches.is_present("print") {
                println!("A: {}", a);
                println!("x0: {}", x);
                println!("b: {}", b);
            }

            match solve_matches.subcommand() {
                ("test", Some(test_matches)) => {
                    use rand::{thread_rng, Rng};
                    use rayon::prelude::*;

                    let method = match test_matches.value_of("algorithm") {
                        Some("gauss-seidel") => solve::IterativeMethod::GaussSeidel,
                        Some("jacobi") => solve::IterativeMethod::Jacobi,
                        _ => unimplemented!(),
                    };
                    let iter_count = value_t!(test_matches, "iters", usize).unwrap();
                    let size_range = {
                        let values = values_t!(test_matches, "size", Index).unwrap();
                        (values[0], values[1])
                    };
                    let value_range = {
                        let values = values_t!(test_matches, "range", f64).unwrap();
                        (values[0], values[1])
                    };
                    let omega = value_t!(test_matches, "omega", f64).unwrap_or(1.0);

                    (size_range.0..size_range.1).into_par_iter()
                    .map(|size| {
                        let a = {
                            let mut a = SparseMatrix::new(Index2 {
                                row: size,
                                col: size
                            });
                            for i in 0..size {
                                a.add(Index2 { row: 0,        col: i }, -1.0);
                                a.add(Index2 { row: size - 1, col: i }, -1.0);
                                a.add(Index2 { row: i,        col: i },  4.0);
                            }
                            a
                        };

                        // println!("a {:?}", a);
                        let mut x = unsafe {
                            let mut v = Vector::new_uninitialized(size);
                            for i in 0..size {
                                v[i] = 1.0;
                            }
                            v
                        };

                        let b = unsafe {
                            let mut v = Vector::new_uninitialized(size);
                            let gen_rand = || thread_rng().gen_range(value_range.0, value_range.1);

                            let first_rand = gen_rand();
                            let last_rand = gen_rand();
                            v[0] = first_rand * 4.0 + last_rand * (-1.0);
                            v[size - 1] = first_rand * (-1.0) + last_rand * (4.0);
                            for i in 1..size - 1 {
                                v[i] = first_rand * (-1.0) + gen_rand() * 4.0 + last_rand * (-1.0);
                            }
                            v
                        };

                        {
                            let mut m = solve::IterativeSolver::build(&a, &mut x, &b)
                                .omega(omega)
                                .method(method)
                                .unwrap().unwrap();

                            (0..iter_count).for_each(|_| {
                                m.next();
                             });

                            (size, m.calc_q().unwrap_or(std::f64::NAN))
                        }

                    })
                    .collect::<Vec<_>>().into_iter()
                    .for_each(|(size, q)| println!("s={} q={:?}", size, q));

                },
                ("iterative", Some(iter_matches)) => {

                    let iter_count = value_t!(iter_matches, "iters", usize).unwrap_or(100);
                    // TODO: find a better name
                    let div_limit = value_t!(iter_matches, "stop", usize).unwrap_or(10);
                    let omega = value_t!(iter_matches, "omega", f64).unwrap_or(1.0);

                    let result = solve::IterativeSolver::build(&a, &mut x, &b)
                        .omega(omega)
                        .method({
                            match iter_matches.value_of("algorithm") {
                                Some("gauss-seidel") => solve::IterativeMethod::GaussSeidel,
                                Some("jacobi") => solve::IterativeMethod::Jacobi,
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
                },
                _ => {}
            }
        },
        _ => {}
    }

}

fn read_square_matrix(path: &str) -> Result<SquareMatrix, ()> {
    let mut lines = read_lines(path);
    let size: Index = lines.next().unwrap().parse().expect("matrix size");

    let mut a = SquareMatrix::new(Index2 { row: size, col: size });

    parse_elements(&mut lines, |(pos, value)| a[pos] = value);

    Ok(a)
}

fn read_combo(path: &str) -> Result<(SparseMatrix, Vector, Vector), ()> {
    let mut lines = read_lines(path);
    let size: Index = lines.next().unwrap().parse().expect("matrix size");

    let mut a = SparseMatrix::new(Index2 {
        row: size,
        col: size
    });

    parse_elements(&mut lines, |(pos, value)| a.add(pos, value));

    let b = unsafe {
        let mut v = Vector::new_uninitialized(size);
        parse_vector(&mut lines, &mut v);
        v
    };

    let x = unsafe {
        let mut v = Vector::new_uninitialized(size);
        parse_vector(&mut lines, &mut v);
        v
    };


    Ok((a, x, b))
}

fn read_lines(path: &str) -> impl Iterator<Item=String> {
    let file = match path {
        "-" => Box::new(io::stdin()) as Box<dyn io::Read>,
        _ => {
            let file = File::open(path).expect("open file");
            Box::new(file) as Box<dyn io::Read>
        }
    };
    BufReader::new(file).lines().scan((), |_, line| line.ok())
}

fn parse_vector<Iter: Iterator<Item=String>>(iter: &mut Iter, v: &mut Vector) {
    (0..v.shape()).for_each(|i| {
        v[i] = iter.next().unwrap().parse().expect("scalar value")
    });
}

fn parse_elements<Iter: Iterator<Item=String>, F: FnMut((Index2, Scalar))>(lines: &mut Iter, mut f: F) {
    let m: Index = lines.next().unwrap().parse().expect("element count");
    (0..m).for_each(|_| {
        let line = lines.next().unwrap();
        let mut tokens = line.split_whitespace();
        let pos = Index2 {
            row: tokens.next().and_then(|t| t.parse::<Index>().ok()).expect("row index") - 1,
            col: tokens.next().and_then(|t| t.parse::<Index>().ok()).expect("column index") - 1,
        };
        let value: Scalar = tokens.next().and_then(|t| t.parse().ok()).expect("scalar value for `a' matrix");

        f((pos, value));
    })
}
