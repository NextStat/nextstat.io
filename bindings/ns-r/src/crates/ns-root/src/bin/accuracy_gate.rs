use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ns_zstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};
use ns_zstd::decoding::{BlockDecodingStrategy, FrameDecoder};

const BUF_SIZE: usize = 64 * 1024;

fn main() -> Result<(), io::Error> {
    let cfg = Config::parse()?;

    let mut files = Vec::<PathBuf>::new();
    for p in &cfg.inputs {
        collect_inputs(p, &mut files)?;
    }
    files.sort();
    files.dedup();

    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no .zst/.zstd files found in inputs",
        ));
    }

    let mut ok = 0usize;
    let mut failed = 0usize;

    for path in files {
        match compare_file(&cfg, &path) {
            Ok(stats) => {
                ok += 1;
                println!("OK   {}  ({} bytes)", path.display(), stats.bytes);
            }
            Err(e) => {
                failed += 1;
                eprintln!("FAIL {}  ({})", path.display(), e);
                if !cfg.keep_going {
                    break;
                }
            }
        }
    }

    println!("\nSummary: {} ok, {} failed (keep_going={})", ok, failed, cfg.keep_going);

    if failed == 0 { Ok(()) } else { Err(io::Error::other("accuracy gate failed")) }
}

#[derive(Clone, Debug)]
struct Config {
    zstd_bin: String,
    inputs: Vec<PathBuf>,
    keep_going: bool,
    verbose: bool,
}

impl Config {
    fn parse() -> Result<Self, io::Error> {
        let mut args = std::env::args_os();
        let _exe = args.next();

        let mut zstd_bin: Option<String> = None;
        let mut keep_going = true;
        let mut verbose = false;
        let mut inputs: Vec<PathBuf> = Vec::new();

        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--zstd" => {
                    let Some(p) = args.next() else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "--zstd requires a value",
                        ));
                    };
                    zstd_bin = Some(p.to_string_lossy().to_string());
                }
                "--fail-fast" => keep_going = false,
                "--keep-going" => keep_going = true,
                "--verbose" | "-v" => verbose = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                s if s.starts_with('-') => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown flag: {s}"),
                    ));
                }
                _ => inputs.push(PathBuf::from(arg)),
            }
        }

        if inputs.is_empty() {
            print_help();
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "missing input paths"));
        }

        let zstd_bin = zstd_bin.unwrap_or_else(|| "zstd".to_string());

        Ok(Self { zstd_bin, inputs, keep_going, verbose })
    }
}

fn print_help() {
    eprintln!(
        "Usage:\n  accuracy_gate [--zstd <path>] [--fail-fast] [--verbose] <file-or-dir>...\n\nBehavior:\n  - Recursively finds .zst/.zstd files\n  - Decompresses with reference libzstd CLI (zstd -dc) and ns-zstd\n  - Compares output byte-for-byte in a streaming fashion\n\nExit status:\n  - 0 if all files match\n  - non-zero on any mismatch\n"
    );
}

fn collect_inputs(p: &Path, out: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    let md = std::fs::metadata(p)?;
    if md.is_dir() {
        collect_dir(p, out)
    } else {
        if is_zstd_file(p) {
            out.push(p.to_path_buf());
        }
        Ok(())
    }
}

fn collect_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), io::Error> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let md = entry.metadata()?;
        if md.is_dir() {
            collect_dir(&path, out)?;
        } else if md.is_file() && is_zstd_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_zstd_file(p: &Path) -> bool {
    matches!(p.extension().and_then(|s| s.to_str()), Some("zst") | Some("zstd"))
}

#[derive(Copy, Clone, Debug)]
struct Stats {
    bytes: u64,
}

fn compare_file(cfg: &Config, path: &Path) -> Result<Stats, io::Error> {
    let mut child = Command::new(&cfg.zstd_bin)
        .arg("-dc")
        .arg("--")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("failed to spawn reference decoder '{}': {e}", cfg.zstd_bin),
            )
        })?;

    let zstd_stdout = child.stdout.take().ok_or_else(|| io::Error::other("missing zstd stdout"))?;
    let zstd_stderr = child.stderr.take().ok_or_else(|| io::Error::other("missing zstd stderr"))?;

    let mut zstd_out = BufReader::new(zstd_stdout);
    let mut zstd_err = BufReader::new(zstd_stderr);

    let ruz_file = File::open(path)?;
    let mut ruz_out = BufReader::new(NsZstdMultiFrameReader::new(ruz_file));

    let bytes = stream_compare(cfg, &mut ruz_out, &mut zstd_out, path)?;

    let status = child.wait()?;
    if !status.success() {
        let mut err_text = String::new();
        zstd_err.read_to_string(&mut err_text)?;
        let err_text = err_text.trim();
        let tail =
            if err_text.is_empty() { "(no stderr)".to_string() } else { err_text.to_string() };
        return Err(io::Error::other(format!(
            "reference zstd failed: status={status}, stderr={tail}"
        )));
    }

    Ok(Stats { bytes })
}

fn stream_compare(
    cfg: &Config,
    a: &mut dyn Read,
    b: &mut dyn Read,
    path: &Path,
) -> Result<u64, io::Error> {
    let mut a_buf = vec![0u8; BUF_SIZE];
    let mut b_buf = vec![0u8; BUF_SIZE];

    let mut a_pos = 0usize;
    let mut a_len = 0usize;
    let mut b_pos = 0usize;
    let mut b_len = 0usize;

    let mut total: u64 = 0;

    loop {
        if a_pos == a_len {
            a_len = a.read(&mut a_buf)?;
            a_pos = 0;
        }
        if b_pos == b_len {
            b_len = b.read(&mut b_buf)?;
            b_pos = 0;
        }

        if a_len == 0 && b_len == 0 {
            return Ok(total);
        }
        if a_len == 0 && b_len != 0 {
            return Err(io::Error::other(format!(
                "output length mismatch at offset {total}: ns-zstd ended early, reference has more data"
            )));
        }
        if b_len == 0 && a_len != 0 {
            return Err(io::Error::other(format!(
                "output length mismatch at offset {total}: reference ended early, ns-zstd has more data"
            )));
        }

        let a_avail = a_len - a_pos;
        let b_avail = b_len - b_pos;
        let n = a_avail.min(b_avail);

        let a_slice = &a_buf[a_pos..a_pos + n];
        let b_slice = &b_buf[b_pos..b_pos + n];

        if a_slice != b_slice {
            let mut i = 0usize;
            while i < n {
                if a_slice[i] != b_slice[i] {
                    break;
                }
                i += 1;
            }
            let off = total + i as u64;

            let a_tail = hex_preview(&a_slice[i..], 32);
            let b_tail = hex_preview(&b_slice[i..], 32);

            let msg = format!(
                "byte mismatch at offset {off}: ns-zstd=0x{:02X} ref=0x{:02X}; ns-zstd_tail={a_tail} ref_tail={b_tail}",
                a_slice[i], b_slice[i]
            );
            if cfg.verbose {
                eprintln!("Mismatch in {}: {msg}", path.display());
            }
            return Err(io::Error::other(msg));
        }

        a_pos += n;
        b_pos += n;
        total += n as u64;
    }
}

fn hex_preview(bytes: &[u8], max: usize) -> String {
    let mut out = String::new();
    let n = bytes.len().min(max);
    for (i, &b) in bytes[..n].iter().enumerate() {
        if i != 0 {
            out.push(' ');
        }
        out.push_str(&format!("{:02X}", b));
    }
    out
}

struct NsZstdMultiFrameReader<R: Read> {
    source: R,
    decoder: FrameDecoder,
    done: bool,
}

impl<R: Read> NsZstdMultiFrameReader<R> {
    fn new(source: R) -> Self {
        Self { source, decoder: FrameDecoder::new(), done: false }
    }

    fn skip_bytes(&mut self, mut to_skip: u64) -> Result<(), io::Error> {
        let mut buf = [0u8; 16 * 1024];
        while to_skip > 0 {
            let want = (to_skip as usize).min(buf.len());
            let got = self.source.read(&mut buf[..want])?;
            if got == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF while skipping skippable frame",
                ));
            }
            to_skip -= got as u64;
        }
        Ok(())
    }
}

impl<R: Read> Read for NsZstdMultiFrameReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        if self.done {
            return Ok(0);
        }

        loop {
            while self.decoder.is_finished() && self.decoder.can_collect() == 0 {
                match self.decoder.init(&mut self.source) {
                    Ok(()) => break,
                    Err(FrameDecoderError::ReadFrameHeaderError(
                        ReadFrameHeaderError::SkipFrame { length, .. },
                    )) => {
                        self.skip_bytes(length as u64)?;
                        continue;
                    }
                    Err(FrameDecoderError::ReadFrameHeaderError(
                        ReadFrameHeaderError::MagicNumberReadError(e),
                    )) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        self.done = true;
                        return Ok(0);
                    }
                    Err(e) => {
                        return Err(io::Error::other(format!(
                            "ns-zstd failed to initialize next frame: {e}"
                        )));
                    }
                }
            }

            while self.decoder.can_collect() < buf.len() && !self.decoder.is_finished() {
                let need = buf.len() - self.decoder.can_collect();
                self.decoder
                    .decode_blocks(&mut self.source, BlockDecodingStrategy::UptoBytes(need))
                    .map_err(|e| io::Error::other(format!("ns-zstd decode_blocks failed: {e}")))?;
            }

            let n = self
                .decoder
                .read(buf)
                .map_err(|e| io::Error::other(format!("ns-zstd read failed: {e}")))?;
            if n != 0 {
                return Ok(n);
            }

            if self.decoder.is_finished() && self.decoder.can_collect() == 0 {
                continue;
            }

            self.decoder
                .decode_blocks(
                    &mut self.source,
                    BlockDecodingStrategy::UptoBytes(buf.len().max(1024)),
                )
                .map_err(|e| io::Error::other(format!("ns-zstd decode_blocks failed: {e}")))?;
        }
    }
}
