#![no_main]
use libfuzzer_sys::fuzz_target;

use vectora::Vector;

fuzz_target!(|number: f64| {
    let _x: Vector<f64, 3> =
        Vector::from([number, number, number]) + Vector::from([number, number, number]);
});
