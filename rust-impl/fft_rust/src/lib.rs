use num_complex::Complex64;
use std::f64::consts::PI as PI64;
use itertools::Itertools;

fn reverse_bits(n: usize, n_bits: usize) -> usize {
    (0..n_bits)
        .map(|i| (n >> i) & 0x01)
        .enumerate()
        .fold(0, |acc, (i, x)| (x << (n_bits - 1 - i)) | acc)
}

fn array_reorder<T: Copy>(arr: &[T]) -> Result<Vec<T>, &str> {
    let n_bits = (arr.len() as f32).log2() as usize;
    if 2 << (n_bits - 1) != arr.len() {
        return Err("len of the array must be 2 ** n.");
    }
    let indexes = (0..arr.len()).map(|i| reverse_bits(i, n_bits));
    Ok(indexes.map(|i| arr[i]).collect())
}

fn get_freqs(n: usize, fs: f64) -> Vec<f64> {
    let df = fs / (n as f64 - 1.);
    (0..n).map(|x| x as f64 * df).collect()
}

fn fft(data: &mut [Complex64]) -> Vec<Complex64> {
    let data = array_reorder(data).unwrap();
    let n_data = data.len();
    let log2n = if n_data == 0 { 0 } else {(n_data as f64).log2() as usize};
    /*
    懒得写了，先放这儿了
    */
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        let x0 = 0b00010;
        let x1 = 0b01000;
        let x2 = reverse_bits(x0, 5);
        assert_eq!(x1, x2);

        let x0 = 0b01;
        let x1 = 0b10;
        let x2 = reverse_bits(x0, 2);
        assert_eq!(x1, x2);

        let x0 = 0b01;
        let x1 = 0b01;
        let x2 = reverse_bits(x0, 1);
        assert_eq!(x1, x2);
    }

    #[test]
    fn test_array_reorder() {
        let arr0: Vec<i32> = (0..8).collect();
        let arr1 = array_reorder(&arr0).unwrap();
        assert_eq!(vec![0, 4, 2, 6, 1, 5, 3, 7], arr1);

        let arr0: Vec<Complex64> = (0..8).map(|x| Complex64::new(x as f64, x as f64)).collect();
        let arr1 = array_reorder(&arr0).unwrap();
        let arr2: Vec<Complex64> = [0, 4, 2, 6, 1, 5, 3, 7].iter().map(|x| Complex64::new(*x as f64, *x as f64)).collect();
        assert_eq!(arr1, arr2);
    }
}
