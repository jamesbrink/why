// Classic Rust borrow checker error - use after move

fn main() {
    let data = vec![1, 2, 3, 4, 5];

    // This moves ownership of data to processed
    let processed = process_data(data);

    // Error! data was moved above, can't use it here
    println!("Original length: {}", data.len());
    println!("Processed: {:?}", processed);
}

fn process_data(input: Vec<i32>) -> Vec<i32> {
    input.into_iter().map(|x| x * 2).collect()
}
