#![allow(unused_imports)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use vectora::types::vector::*;

// ================================
//
// Vector Initialization
//
// ================================

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// Zero vectors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: i32 zero", |b| b.iter(|| Vector::<i32, 3>::zero()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: f64 zero", |b| b.iter(|| Vector::<f64, 3>::zero()));
// }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// i32 3D Vectors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: i32 3D from array", |b| {
//         b.iter(|| Vector::<i32, 3>::from([1, 2, 3]))
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: i32 3D from iterator over array", |b| {
//         b.iter(|| [1, 2, 3].into_iter().collect::<Vector<i32, 3>>())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: i32 3D from iterator over Vec", |b| {
//         b.iter(|| vec![1, 2, 3].into_iter().collect::<Vector<i32, 3>>())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let arr = [1, 2, 3];
//     let sli = &arr[..];
//     c.bench_function("Vector Initialization: i32 3D from array slice", |b| {
//         b.iter(|| Vector::<i32, 3>::try_from(sli).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let vec = vec![1, 2, 3];
//     let sli = &vec[..];
//     c.bench_function("Vector Initialization: i32 3D from Vec slice", |b| {
//         b.iter(|| Vector::<i32, 3>::try_from(sli).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let vec = vec![1, 2, 3];
//     c.bench_function("Vector Initialization: i32 3D from Vec ref", |b| {
//         b.iter(|| Vector::<i32, 3>::try_from(&vec).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     fn init_with_vec() {
//         let vect = vec![1, 2, 3];
//         let _: Vector<i32, 3> = Vector::try_from(black_box(vect)).unwrap();
//     }

//     c.bench_function("Vector Initialization: i32 3D from Vec", move |b| b.iter(|| init_with_vec()));
// }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// f64 3D Vectors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: f64 3D from array", |b| {
//         b.iter(|| Vector::<f64, 3>::from([1.0, 2.0, 3.0]))
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: f64 3D from iterator over array", |b| {
//         b.iter(|| [1.0, 2.0, 3.0].into_iter().collect::<Vector<f64, 3>>())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     c.bench_function("Vector Initialization: f64 3D from iterator over Vec", |b| {
//         b.iter(|| vec![1.0, 2.0, 3.0].into_iter().collect::<Vector<f64, 3>>())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let arr = [1.0_f64, 2.0_f64, 3.0_f64];
//     let sli = &arr[..];
//     c.bench_function("Vector Initialization: f64 3D from array slice", |b| {
//         b.iter(|| Vector::<f64, 3>::try_from(sli).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let vec = vec![1.0_f64, 2.0_f64, 3.0_f64];
//     let sli = &vec[..];
//     c.bench_function("Vector Initialization: f64 3D from Vec slice", |b| {
//         b.iter(|| Vector::<f64, 3>::try_from(sli).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let vec = vec![1.0_f64, 2.0_f64, 3.0_f64];
//     c.bench_function("Vector Initialization: f64 3D from Vec ref", |b| {
//         b.iter(|| Vector::<f64, 3>::try_from(&vec).unwrap())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     fn init_with_vec() {
//         let vect = vec![1.0_f64, 2.0_f64, 3.0_f64];
//         let _: Vector<f64, 3> = Vector::try_from(black_box(vect)).unwrap();
//     }

//     c.bench_function("Vector Initialization: f64 3D from Vec", move |b| b.iter(|| init_with_vec()));
// }

// ================================
//
// Partial Eq relation testing
//
// ================================

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// i32 3D Vectors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

// fn criterion_benchmark(c: &mut Criterion) {
//     let v1: Vector<i32, 3> = Vector::from([1, 2, 3]);
//     let v2: Vector<i32, 3> = Vector::from([1, 2, 3]);
//     c.bench_function("PartialEq testing: i32 3D Vector : i32 3D Vector", |b| b.iter(|| v1 == v2));
// }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// f64 3D Vectors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

// fn criterion_benchmark(c: &mut Criterion) {
//     let v1: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//     let v2: Vector<f64, 3> = Vector::from([1.0, 2.0, 3.0]);
//     c.bench_function("PartialEq testing: f64 3D Vector : f64 3D Vector", |b| b.iter(|| v1 == v2));
// }

// ================================
//
// Method impl
//
// ================================

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<i32, 3> = Vector::from([1, -2, 3]);
//     c.bench_function("Vector product method: i32 3D Vector", |b| b.iter(|| v.product()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 3> = Vector::from([1.0, -2.0, 3.0]);
//     c.bench_function("Vector product method: f64 3D Vector", |b| b.iter(|| v.product()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f32, 5> = Vector::from([1.0, -2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean method: f32 5D Vector", |b| b.iter(|| v.mean()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 5> = Vector::from([1.0, -2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean method: f64 5D Vector", |b| b.iter(|| v.mean()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f32, 5> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean_geo method: f32 5D Vector", |b| b.iter(|| v.mean_geo()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 5> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean_geo method: f64 5D Vector", |b| b.iter(|| v.mean_geo()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f32, 5> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean_harmonic method: f32 5D Vector", |b| {
//         b.iter(|| v.mean_harmonic())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 5> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0]);
//     c.bench_function("Vector mean_harmonic method: f64 5D Vector", |b| {
//         b.iter(|| v.mean_harmonic())
//     });
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f32, 10> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0, 9.0, 72.0, 1.0, 52.0, 88.0]);
//     c.bench_function("Vector median method (even): f32 10D Vector", |b| b.iter(|| v.median()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f32, 9> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0, 9.0, 72.0, 1.0, 52.0]);
//     c.bench_function("Vector median method (odd): f32 10D Vector", |b| b.iter(|| v.median()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 10> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0, 9.0, 72.0, 1.0, 52.0, 88.0]);
//     c.bench_function("Vector median method (even): f64 10D Vector", |b| b.iter(|| v.median()));
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     let v: Vector<f64, 9> = Vector::from([1.0, 2.0, 3.0, 100.0, 24.0, 9.0, 72.0, 1.0, 52.0]);
//     c.bench_function("Vector median method (odd): f64 10D Vector", |b| b.iter(|| v.median()));
// }

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
