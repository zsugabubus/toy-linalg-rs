use rayon::prelude::*;
use std::{ops, mem, fmt};
use std::iter::Iterator;
use std::ops::Range;
use std::marker::PhantomData;

pub type Index = u32;
/// Vector space coordinates in (_col_, _row_) format
pub type Index2 = (Index, Index);
pub type Scalar = f64; // shitty workaround...

////////////////////////////////////////////////////////////////////////

pub enum Method {
    Jacobi,
    GaussSeidel
}

////////////////////////////////////////////////////////////////////////

pub struct IterativeMethod<'a> {
    a: &'a SparseMatrix,
    x: (&'a mut Vector, &'a mut Vector),
    b: &'a Vector,
    norm: (Scalar, Scalar),
    omega: f64,
    method: Method,
    our_x: bool,
    _lifetime: &'a PhantomData<()>,
}

impl<'a> IterativeMethod<'a> {
    pub fn build(a: &'a SparseMatrix, x: &'a mut Vector, b: &'a Vector) -> IterativeMethodBuilder<'a> {
        IterativeMethodBuilder::new(a, x, b)
    }

    pub fn calc_q(&self) -> Option<Scalar> {
        match self.norm.0 {
            q if q.is_infinite() => None,
            _ => Some((self.norm.0 * self.norm.1).sqrt()),
        }
    }
}

impl<'a> Iterator for IterativeMethod<'a> {
    type Item = Option<Scalar>;

    fn next(&mut self) -> Option<Self::Item> {

        let norm = match self.method {
            Method::GaussSeidel => {
                (0..self.a.shape().1).into_iter()
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
            Method::Jacobi => {
                (0..self.a.shape().1).into_par_iter()
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

impl<'a> Drop for IterativeMethod<'a> {
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

pub struct IterativeMethodBuilder<'a> {
    a: &'a SparseMatrix,
    x: &'a mut Vector,
    b: &'a Vector,
    omega: f64,
    method: Option<Method>,
    _lifetime: &'a PhantomData<u8>,
}

impl<'a> IterativeMethodBuilder<'a> {
    pub fn new(a: &'a SparseMatrix, x: &'a mut Vector, b: &'a Vector) -> IterativeMethodBuilder<'a> {
        IterativeMethodBuilder {
            a,
            x,
            b,
            omega: 1.0,
            method: None,
            _lifetime: &PhantomData
        }
    }

    pub fn omega(mut self, omega: f64) -> IterativeMethodBuilder<'a> {
        self.omega = omega;
        self
    }

    pub fn method(mut self, method: Method) -> IterativeMethodBuilder<'a> {
        self.method = Some(method);
        self
    }

    pub fn unwrap(self) -> Result<IterativeMethod<'a>, ()> {
        let size = self.a.shape().0;
        Ok(IterativeMethod {
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
                v.resize(shape.1 as usize, (Index::max_value(), 0));
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
        (self.elems.len(), (self.shape.0 * self.shape.1) as usize)
    }

    pub fn add(&mut self, pos: Index2, value: Scalar) {
        if value != Default::default() {
            let range = self.row(pos.1).range();
            let start = range.start;
            match self.col_indexes[range].binary_search(&pos.0) {
                Err(index) => {
                    let index = index + start;
                    self.elems.insert(index, value);
                    self.col_indexes.insert(index, pos.0);
                    let marks = &mut self.row_marks[pos.1 as usize];
                    if pos.0 >= pos.1 && (marks.0 == Index::max_value() || self.col_indexes[marks.0 as usize] > pos.0) {
                        marks.0 = index as Index;
                    } else if (index as Index) <= marks.0 && marks.0 < Index::max_value() {
                        marks.0 = marks.0.wrapping_add(1);
                    }

                    marks.1 = marks.1.wrapping_add(1);
                    (&mut self.row_marks[pos.1 as usize + 1..])
                        .iter_mut()
                        .for_each(|mut m| {
                            if m.0 < Index::max_value() {
                                m.0 = m.0.wrapping_add(1);
                            }
                            m.1 = m.1.wrapping_add(1);
                        });
                    // println!("{:?}", self);
                },
                Ok(index) => {
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
        for row in 0..self.shape.1 {
            write!(f, " {:>5?}:", row)?;
            for (col, value) in self.row(row).into_iter() {
                write!(f, " ({}) {:20.10e}", col, value)?;
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
            write!(f, " {:20.10e}\n", self[row])?;
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
pub struct Matrix {
    shape: Index2,
    elems: Box<[Scalar]>
}

impl Matrix {
    pub fn new(shape: Index2) -> Matrix {
        Matrix {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize((shape.0 * shape.1) as usize, Default::default());
                v.into_boxed_slice()
            }
        }
    }

    pub unsafe fn new_uninitialized(shape: Index2) -> Matrix {
        Matrix {
            shape,
            elems: {
                let mut v = Vec::with_capacity(0);
                v.resize((shape.0 * shape.1) as usize, mem::uninitialized());
                v.into_boxed_slice()
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[[\n")?;
        for row in 0..self.shape.1 {
            for col in 0..self.shape.0 {
                write!(f, " {:20.10e}", self[(col, row)])?;
            }
            write!(f, "\n")?;
        }
        write!(f, "]]")
    }
}

impl ops::Index<Index2> for Matrix {
    type Output = Scalar;

    #[inline]
    fn index(&self, index: Index2) -> &Self::Output {
        &self.elems[(index.0 + self.shape.0 * index.1) as usize]
    }
}

impl ops::IndexMut<Index2> for Matrix {
    #[inline]
    fn index_mut(&mut self, index: Index2) -> &mut Self::Output {
        &mut self.elems[(index.0 + self.shape.0 * index.1) as usize]
    }
}

// vim: expandtab ts=4 sw=4
