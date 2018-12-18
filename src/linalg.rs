use rayon::prelude::*;
use std::marker::PhantomData;
use std::{ops, mem, fmt, ptr};
use std::ops::Range;

pub type Index = u32;

#[derive(Clone, Copy, Debug)]
pub struct Index2 {
    pub row: Index,
    pub col: Index,
}

pub type Scalar = f64; // shitty workaround...

impl fmt::Display for Index2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

////////////////////////////////////////////////////////////////////////

pub mod solve {
    use super::*;

    #[derive(Clone, Copy)]
    pub enum IterativeMethod {
        Jacobi,
        GaussSeidel
    }

    ////////////////////////////////////////////////////////////////////////

    pub struct IterativeSolver<'a> {
        a: &'a SparseMatrix,
        x: (&'a mut Vector, &'a mut Vector),
        b: &'a Vector,
        norm: (Scalar, Scalar),
        omega: f64,
        method: IterativeMethod,
        our_x: bool,
        _lifetime: &'a PhantomData<()>,
    }

    impl<'a> IterativeSolver<'a> {
        pub fn build(a: &'a SparseMatrix, x: &'a mut Vector, b: &'a Vector) -> IterativeSolverBuilder<'a> {
            IterativeSolverBuilder::new(a, x, b)
        }

        pub fn calc_q(&self) -> Option<Scalar> {
            match self.norm.0 {
                q if q.is_infinite() => None,
                _ => Some((self.norm.0 * self.norm.1).sqrt()),
            }
        }
    }

    impl<'a> Iterator for IterativeSolver<'a> {
        type Item = Option<Scalar>;

        fn next(&mut self) -> Option<Self::Item> {

            let norm = match self.method {
                IterativeMethod::GaussSeidel => {
                    (0..self.a.shape().row).into_iter()
                        .map(|row_index| {
                            let sparse_row = self.a.row(row_index);
                            let (left, diag, right) = sparse_row.into_parts();

                            let next_value = left.fold(-self.b[row_index], |prod_sum, (col, value)| {
                                self.x.1[col].mul_add(value, prod_sum)
                            });
                            let next_value = right.fold(next_value, |prod_sum, (col, value)| {
                                self.x.0[col].mul_add(value, prod_sum)
                            });
                            let next_value = -next_value / diag;
                            let prev_value = self.x.0[row_index];

                            #[allow(mutable_transmutes)]
                            let new_value = unsafe {
                                let new_value = mem::transmute::<&Scalar, &mut Scalar>(&self.x.1[row_index]);
                                *new_value = self.omega * next_value + (1.0 - self.omega) * prev_value;
                                *new_value
                            };

                            let delta = new_value - prev_value;

                            delta.abs()
                        })
                        .sum()
                },
                IterativeMethod::Jacobi => {
                    (0..self.a.shape().row).into_par_iter()
                        .map(|row_index| {
                            let sparse_row = self.a.row(row_index);
                            let d = sparse_row.diagonal();

                            let prod_sum = sparse_row.into_iter()
                                .fold(-self.x.0[row_index] * d - self.b[row_index], |prod_sum, (col, value)| {
                                    self.x.0[col].mul_add(value, prod_sum)
                                });
                            let next_value = -prod_sum / d;
                            let prev_value = self.x.0[row_index];

                            #[allow(mutable_transmutes)]
                            let new_value = unsafe {
                                let new_value = mem::transmute::<&Scalar, &mut Scalar>(&self.x.1[row_index]);
                                *new_value = self.omega * next_value + (1.0 - self.omega) * prev_value;
                                *new_value
                            };

                            let delta = new_value - prev_value;

                            delta.abs()
                        })
                        .sum()
                },
            };

            self.norm = (self.norm.1, norm);

            // new values are in x_1 vector, move it to x_0
            mem::swap(&mut self.x.0, &mut self.x.1);
            self.our_x = !self.our_x;

            Some(self.calc_q())
        }
    }

    impl<'a> Drop for IterativeSolver<'a> {
        fn drop(&mut self) {
            unsafe {
                Box::from_raw({
                    if self.our_x {
                        self.x.1
                    } else {
                        self.x.0
                    }
                })
            };
        }
    }

    ////////////////////////////////////////////////////////////////////////

    pub struct IterativeSolverBuilder<'a> {
        a: &'a SparseMatrix,
        x: &'a mut Vector,
        b: &'a Vector,
        omega: f64,
        method: Option<IterativeMethod>,
        _lifetime: &'a PhantomData<u8>,
    }

    impl<'a> IterativeSolverBuilder<'a> {
        pub fn new(a: &'a SparseMatrix, x: &'a mut Vector, b: &'a Vector) -> IterativeSolverBuilder<'a> {
            IterativeSolverBuilder {
                a,
                x,
                b,
                omega: 1.0,
                method: None,
                _lifetime: &PhantomData
            }
        }

        pub fn omega(mut self, omega: f64) -> IterativeSolverBuilder<'a> {
            self.omega = omega;
            self
        }

        pub fn method(mut self, method: IterativeMethod) -> IterativeSolverBuilder<'a> {
            self.method = Some(method);
            self
        }

        pub fn unwrap(self) -> Result<IterativeSolver<'a>, ()> {
            let size = self.a.shape.col;
            Ok(IterativeSolver {
                a: self.a,
                x: {
                    if self.x.shape() != size {
                        return Err(());
                    }

                    let back_vec = Box::leak(
                        Box::new(unsafe { Vector::new_uninitialized(size) })
                    );
                    (self.x, back_vec)
                },
                b: {
                    if self.b.shape() != size {
                        return Err(());
                    }
                    self.b
                },
                omega: self.omega,
                method: self.method.expect("no method specified"),
                our_x: true,
                norm: (unsafe { mem::uninitialized() }, std::f64::INFINITY),
                _lifetime: &PhantomData,
            })
        }
    }

}

////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct SparseMatrix {
    /// Shape of matrix
    shape: Index2,
    /// Non-zero elements
    elems: Vec<Scalar>,
    /// Column index of elements
    col_indexes: Vec<Index>,
    /// Indexes of the middle and the last elements of rows
    ///
    /// Note: If the row doesn't have non-zero diagonal element, it
    /// will store the index of the next non-zero element in the same
    /// row, for performance reasons.
    row_marks: Vec<(Index, Index)>,
}

impl SparseMatrix {
    pub fn new(shape: Index2) -> SparseMatrix {
        SparseMatrix {
            shape,
            elems: Vec::with_capacity(0),
            col_indexes: Vec::with_capacity(0),
            row_marks: {
                let mut v = Vec::with_capacity(0);
                v.resize(shape.row as usize, (Index::max_value(), 0));
                v
            }
        }
    }

    #[inline]
    pub fn row(&self, row: Index) -> Row {
        Row { matrix: &self, row }
    }

    #[inline]
    pub fn shape(&self) -> Index2 {
        self.shape
    }

    pub fn density(&self) -> (usize, usize) {
        (self.elems.len(), (self.shape.col * self.shape.row) as usize)
    }

    pub fn add(&mut self, pos: Index2, value: Scalar) {
        if pos.col >= self.shape.col {
            panic!("index out of bounds: the len is {} but index is {}", self.shape.col, pos.col);
        } else if pos.row >= self.shape.row {
            panic!("index out of bounds: the len is {} but index is {}", self.shape.row, pos.row);
        }

        if value != Default::default() {
            let range = self.row(pos.row).range();
            let start = range.start;
            match self.col_indexes[range].binary_search(&pos.col) {
                Err(rel_index) => {
                    let index = rel_index + start;
                    let marks = &mut self.row_marks[pos.row as usize];
                    if pos.col >= pos.row && (marks.0 == Index::max_value() || self.col_indexes[marks.0 as usize] > pos.col) {
                        marks.0 = index as Index;
                    } else if (index as Index) <= marks.0 && marks.0 < Index::max_value() {
                        marks.0 = marks.0.wrapping_add(1);
                    }

                    marks.1 = marks.1.wrapping_add(1);

                    self.elems.insert(index, value);
                    self.col_indexes.insert(index, pos.col);

                    (&mut self.row_marks[pos.row as usize + 1..])
                        .iter_mut()
                        .for_each(|mut m| {
                            if m.0 < Index::max_value() {
                                m.0 = m.0.wrapping_add(1);
                            }
                            m.1 = m.1.wrapping_add(1);
                        });
                },
                Ok(rel_index) => {
                    let index = rel_index + start;
                    self.elems[index] = value;
                },
            }
        }
    }
}

impl Extend<(Index2, Scalar)> for SparseMatrix {

    fn extend<Iter: IntoIterator<Item=(Index2, Scalar)>>(&mut self, iter: Iter)
    {
        iter.into_iter()
            .for_each(|e| self.add(e.0, e.1));
    }
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[[ \n")?;
        for row in 0..self.shape.row {
            write!(f, " {:>5?}:", row)?;
            for (col, value) in self.row(row).into_iter() {
                write!(f, " [{}] {:12.5e}", col, value)?;
            }
            write!(f, "\n")?;
        }
        write!(f, "]]")
    }
}

////////////////////////////////////////////////////////////////////////

pub struct Row<'a> {
    matrix: &'a SparseMatrix,
    row: Index,
}

impl<'a> Row<'a> {
    #[inline]
    pub fn diagonal(&self) -> Scalar {
        match self.matrix.row_marks[self.row as usize].0 {
            v if v == Index::max_value() => Default::default(),
            v if self.matrix.col_indexes[v as usize] != self.row => Default::default(),
            v => self.matrix.elems[v as usize],
        }
    }

    #[inline]
    pub fn into_parts(self) -> (RowIter<'a>, Scalar, RowIter<'a>) {
        let range = self.range();
        let (left_range, d, right_range) = match self.matrix.row_marks[self.row as usize].0 {
            v if v == Index::max_value() => {
                (0..0, Default::default(), 0..0)
            },
            v if self.matrix.col_indexes[v as usize] != self.row => {
                (range.start..v as usize, Default::default(), v as usize..range.end)
            },
            v => {
                (range.start..v as usize, self.matrix.elems[v as usize], (v + 1) as usize..range.end)
            },
        };

        let left = RowIter { matrix: self.matrix, range: left_range };
        let right = RowIter { matrix: self.matrix, range: right_range };

        (left, d, right)
    }

    #[inline]
    pub fn range(&self) -> Range<usize> {
        let start = if self.row > 0 {
            self.matrix.row_marks[(self.row - 1) as usize].1
        } else {
            0
        } as usize;
        let end = self.matrix.row_marks[self.row as usize].1 as usize;

        start..end
    }
}

impl<'a> IntoIterator for Row<'a> {
    type Item = (Index, Scalar);
    type IntoIter = RowIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        RowIter { matrix: self.matrix, range: self.range() }
    }
}

////////////////////////////////////////////////////////////////////////

pub struct RowIter<'a> {
    matrix: &'a SparseMatrix,
    range: Range<usize>
}

impl<'a> Iterator for RowIter<'a> {
    type Item = (Index, Scalar);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|i| {
            (self.matrix.col_indexes[i as usize], self.matrix.elems[i as usize])
        })
    }
}

////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Vector {
    shape: Index,
    elems: Box<[Scalar]>
}

impl Vector {
    pub fn new(shape: Index) -> Vector {
        Vector {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize(shape as usize, Default::default());
                v.into_boxed_slice()
            }
        }
    }

    pub unsafe fn new_uninitialized(shape: Index) -> Vector {
        Vector {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize(shape as usize, mem::uninitialized());
                v.into_boxed_slice()
            }
        }
    }

    #[inline]
    pub fn shape(&self) -> Index {
        self.shape
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[\n")?;
        for row in 0..self.shape {
            write!(f, " {:12.5e}\n", self[row])?;
        }
        write!(f, "]")
    }
}

impl ops::Index<Index> for Vector {
    type Output = Scalar;

    #[inline]
    fn index(&self, index: Index) -> &Self::Output {
        &self.elems[index as usize]
    }
}

impl ops::IndexMut<Index> for Vector {
    #[inline]
    fn index_mut(&mut self, index: Index) -> &mut Self::Output {
        &mut self.elems[index as usize]
    }
}

////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct SquareMatrix {
    shape: Index,
    elems: Box<[Scalar]>
}

impl SquareMatrix {
    pub fn new(shape: Index2) -> SquareMatrix {
        if shape.row != shape.col {
            panic!("expected equal dimensions but shape is {}", shape);
        }

        SquareMatrix {
            shape: shape.row,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize((shape.col * shape.row) as usize, Default::default());
                v.into_boxed_slice()
            }
        }
    }

    pub unsafe fn new_uninitialized(shape: Index2) -> SquareMatrix {
        if shape.row != shape.col {
            panic!("expected equal dimensions but shape is {}", shape);
        }

        SquareMatrix {
            shape: shape.row,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize((shape.col * shape.row) as usize, mem::uninitialized());
                v.into_boxed_slice()
            }
        }
    }

    pub fn shape(&self) -> Index2 {
        Index2 { row: self.shape, col: self.shape }
    }

    pub fn transpose(&mut self) {
        (0..self.shape).for_each(|i| {
            (i + 1..self.shape).for_each(|j| {
                let ji = Index2 { row: j, col: i };
                let ij = Index2 { row: i, col: j };

                let a = &mut self[ji] as *mut Scalar;
                let b = &mut self[ij] as *mut Scalar;
                unsafe {
                    ptr::swap(a, b);
                }
            });
        });
    }
}

impl fmt::Display for SquareMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[[\n")?;
        for row in 0..self.shape {
            for col in 0..self.shape {
                write!(f, " {:12.5e}", self[Index2 { row, col }])?;
            }
            write!(f, "\n")?;
        }
        write!(f, "]]")
    }
}

impl ops::Index<Index2> for SquareMatrix {
    type Output = Scalar;

    #[inline]
    fn index(&self, index: Index2) -> &Self::Output {
        &self.elems[(index.col + self.shape * index.row) as usize]
    }
}

impl ops::IndexMut<Index2> for SquareMatrix {
    #[inline]
    fn index_mut(&mut self, index: Index2) -> &mut Self::Output {
        &mut self.elems[(index.col + self.shape * index.row) as usize]
    }
}

////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct UpperTriangularMatrix {
    shape: Index,
    elems: Box<[Scalar]>
}

impl UpperTriangularMatrix {
    pub fn new(shape: Index2) -> UpperTriangularMatrix {
        if shape.col != shape.row {
            panic!("expected equal dimensions but shape is {}", shape);
        }

        let shape = shape.row;
        UpperTriangularMatrix {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize(((shape * (shape + 1)) / 2) as usize, Default::default());
                v.into_boxed_slice()
            }
        }
    }

    pub unsafe fn new_uninitialized(shape: Index2) -> UpperTriangularMatrix {
        if shape.col != shape.row {
            panic!("expected equal dimensions but shape is {}", shape);
        }

        let shape = shape.row;
        UpperTriangularMatrix {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize(((shape * (shape + 1)) / 2) as usize, mem::uninitialized());
                v.into_boxed_slice()
            }
        }
    }

    pub fn shape(&self) -> Index2 {
        Index2 { row: self.shape, col: self.shape }
    }

}

impl fmt::Display for UpperTriangularMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[[\n")?;
        (0..self.shape).try_for_each(|row| {
            (0..row).try_for_each(|_col| {
                write!(f, " {:12.5e}", 0.0)
            })?;
            (row..self.shape).try_for_each(|col| {
                write!(f, " {:12.5e}", self[Index2 { row, col }])
            })?;
            write!(f, "\n")
        })?;
        write!(f, "]]")
    }
}

impl ops::Index<Index2> for UpperTriangularMatrix {
    type Output = Scalar;

    #[inline]
    fn index(&self, index: Index2) -> &Self::Output {
        if index.row > index.col
        || index.col >= self.shape
        || index.row >= self.shape {
            panic!("index out of bounds: the shape is {} but index is {}", self.shape(), index);
        }

        &self.elems[(((index.col * (index.col + 1)) / 2) + index.row) as usize]
    }
}

impl ops::IndexMut<Index2> for UpperTriangularMatrix {
    #[inline]
    fn index_mut(&mut self, index: Index2) -> &mut Self::Output {
        if index.row > index.col
        || index.col >= self.shape
        || index.row >= self.shape {
            panic!("index out of bounds: the shape is {} but index is {}", self.shape(), index);
        }

        &mut self.elems[(((index.col * (index.col + 1)) / 2) + index.row) as usize]
    }
}


//////////////////////////////////////////////////////////////////////////////////

pub mod decomp {

    use crate::linalg::*;

    pub enum QRMethod {
        GramSchmidt,
        Householder,
    }

    pub struct QR {
        q: SquareMatrix,
        r: UpperTriangularMatrix,
    }

    impl QR {
        fn gram_schmidt_method(a: &SquareMatrix, q: &mut SquareMatrix, r: &mut UpperTriangularMatrix) {
            // For performance reasons, compute q as a row-vector matrix.

            let size = a.shape().row;

            (0..size).for_each(|k| {
                // copy a_k to q_k
                (0..size).map(|i| {
                    (Index2 { row: k, col: i}, Index2 { row: i, col: k })
                }).for_each(|(iq, ia)| {
                    q[iq] = a[ia]
                });

                // q_k := q_k - r_ik*q_i .. - r_0k*q_0
                (0..k).for_each(|i| {
                    let ri = &mut r[Index2 { row: i, col: k }];
                    // scalar product of q_i and a_k
                    *ri = (0..size).fold(0.0, |sum, j| {
                        let qij = q[Index2 { row: i, col: j }];
                        let akj = q[Index2 { row: k, col: j }];
                        sum + akj * qij
                    });

                    let ri = *ri;
                    (0..size).for_each(|j| {
                        let qij = q[Index2 { row: i, col: j }];
                        let qkj = &mut q[Index2 { row: k, col: j }];
                        *qkj -= ri * qij;
                    });
                });

                // compute length of q_k
                let norm = (0..size).fold(0.0, |sum, i| {
                    let qki = q[Index2 { row: k, col: i}]; // Please, do not pronounce as 'penis'.
                    sum + qki * qki
                }).sqrt();

                r[Index2 { row: k, col: k }] = norm;

                let inv_norm = 1.0 / norm;

                // normalize q_k
                (0..size).for_each(|i| {
                    q[Index2 { row: k, col: i}] *= inv_norm;
                });
            });

            q.transpose();
        }

        pub fn decompose(a: &SquareMatrix, method: QRMethod) -> Self {
            unsafe {
                let mut q = SquareMatrix::new_uninitialized(a.shape());
                let mut r = UpperTriangularMatrix::new_uninitialized(a.shape());

                match method {
                    QRMethod::GramSchmidt => Self::gram_schmidt_method(&a, &mut q, &mut r),
                    QRMethod::Householder => unimplemented!(),
                }

                QR { q, r }
            }
        }

        pub fn into_parts(self) -> (SquareMatrix, UpperTriangularMatrix) {
            (self.q, self.r)
        }
    }

}

// vim: expandtab ts=4 sw=4
