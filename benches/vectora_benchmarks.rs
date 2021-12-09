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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
