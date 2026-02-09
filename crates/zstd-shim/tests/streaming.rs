use std::io::Read as _;
use std::io::Write as _;

struct CountingWriter {
    buf: Vec<u8>,
    writes: usize,
}

impl CountingWriter {
    fn new() -> Self {
        Self { buf: Vec::new(), writes: 0 }
    }
}

impl std::io::Write for CountingWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.writes += 1;
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn shim_streaming_encoder_emits_bytes_before_finish() {
    let input: Vec<u8> = (0u32..150_000u32).flat_map(|x| x.to_le_bytes()).collect();

    let w = CountingWriter::new();
    let mut enc = zstd::stream::write::Encoder::new(w, 0).unwrap();

    // Write a bit and flush. We should have produced some output bytes already.
    enc.write_all(&input[..64 * 1024]).unwrap();
    enc.flush().unwrap();
    assert!(!enc.get_ref().buf.is_empty(), "expected encoder to emit bytes on flush");
    assert!(enc.get_ref().writes > 0, "expected encoder to write to underlying writer");

    // Finish and decode.
    let w = enc.finish().unwrap();
    let mut dec = zstd::stream::read::Decoder::new(std::io::Cursor::new(&w.buf)).unwrap();
    let mut decoded = Vec::new();
    dec.read_to_end(&mut decoded).unwrap();

    // We flushed mid-stream, so the decoder only sees bytes for a single frame if the encoder keeps a
    // single frame open (ns-zstd StreamingEncoder does). Roundtrip must match.
    assert_eq!(decoded, input[..64 * 1024].to_vec());
}

#[test]
fn shim_streaming_encoder_roundtrips_full_input() {
    let input: Vec<u8> = (0u32..250_000u32).flat_map(|x| x.to_le_bytes()).collect();

    let mut out = Vec::new();
    let mut enc = zstd::stream::write::Encoder::new(&mut out, 0).unwrap();
    for chunk in input.chunks(1024) {
        enc.write_all(chunk).unwrap();
    }
    let _w = enc.finish().unwrap();

    let mut dec = zstd::stream::read::Decoder::new(std::io::Cursor::new(&out)).unwrap();
    let mut decoded = Vec::with_capacity(input.len());
    dec.read_to_end(&mut decoded).unwrap();
    assert_eq!(decoded, input);
}
