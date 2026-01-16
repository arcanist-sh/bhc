//! Basel Haskell Compiler Interactive (bhci) - REPL
//!
//! An interactive environment for exploring Haskell 2026 code.

use anyhow::Result;
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("Basel Haskell Compiler Interactive (bhci)");
    println!("Version {}", env!("CARGO_PKG_VERSION"));
    println!("Type :help for help, :quit to exit");
    println!();

    // Initialize keyword interner
    bhc_intern::kw::intern_all();

    let mut line_num = 1;
    let mut input_buffer = String::new();

    loop {
        // Print prompt
        let prompt = if input_buffer.is_empty() {
            format!("bhci:{:03}> ", line_num)
        } else {
            format!("bhci:{:03}| ", line_num)
        };
        print!("{}", prompt);
        io::stdout().flush()?;

        // Read line
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            // EOF
            break;
        }

        let line = line.trim_end();

        // Handle commands
        if line.starts_with(':') {
            match handle_command(line) {
                CommandResult::Continue => {}
                CommandResult::Quit => break,
            }
            line_num += 1;
            continue;
        }

        // Add to buffer
        input_buffer.push_str(line);
        input_buffer.push('\n');

        // Try to parse and evaluate
        // For now, just parse expressions
        let trimmed = input_buffer.trim();
        if !trimmed.is_empty() {
            eval_input(trimmed);
        }

        input_buffer.clear();
        line_num += 1;
    }

    println!("\nGoodbye!");
    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
}

fn handle_command(cmd: &str) -> CommandResult {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let cmd_name = parts.first().map(|s| *s).unwrap_or("");

    match cmd_name {
        ":quit" | ":q" => CommandResult::Quit,
        ":help" | ":h" | ":?" => {
            print_help();
            CommandResult::Continue
        }
        ":type" | ":t" => {
            if parts.len() > 1 {
                let expr = parts[1..].join(" ");
                show_type(&expr);
            } else {
                println!("Usage: :type <expression>");
            }
            CommandResult::Continue
        }
        ":info" | ":i" => {
            if parts.len() > 1 {
                let name = parts[1];
                show_info(name);
            } else {
                println!("Usage: :info <name>");
            }
            CommandResult::Continue
        }
        ":load" | ":l" => {
            if parts.len() > 1 {
                let file = parts[1];
                load_file(file);
            } else {
                println!("Usage: :load <file>");
            }
            CommandResult::Continue
        }
        ":reload" | ":r" => {
            println!("Reloading...");
            // TODO: Reload previously loaded modules
            CommandResult::Continue
        }
        ":browse" | ":b" => {
            println!("No modules loaded");
            CommandResult::Continue
        }
        ":set" => {
            if parts.len() > 1 {
                set_option(&parts[1..].join(" "));
            } else {
                println!("Current settings:");
                println!("  profile: default");
                println!("  edition: 2026");
            }
            CommandResult::Continue
        }
        _ => {
            println!("Unknown command: {}", cmd_name);
            println!("Type :help for help");
            CommandResult::Continue
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  :help, :h, :?     Show this help");
    println!("  :quit, :q         Exit the REPL");
    println!("  :type <expr>      Show the type of an expression");
    println!("  :info <name>      Show information about a name");
    println!("  :load <file>      Load a Haskell module");
    println!("  :reload           Reload the current module");
    println!("  :browse           Browse the current module");
    println!("  :set <option>     Set a REPL option");
    println!();
    println!("Options:");
    println!("  :set profile <default|server|numeric|edge>");
    println!("  :set edition <2026>");
}

fn eval_input(input: &str) {
    use bhc_diagnostics::{DiagnosticRenderer, SourceMap};
    use bhc_parser::parse_expr;
    use bhc_span::FileId;

    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file("<repl>".to_string(), input.to_string());

    let (expr, diagnostics) = parse_expr(input, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&source_map);
        renderer.render_all(&diagnostics);
    }

    if let Some(expr) = expr {
        // TODO: Type check and evaluate
        println!("{:?}", expr);
    }
}

fn show_type(expr: &str) {
    println!("(type inference not yet implemented)");
    println!("{} :: ???", expr);
}

fn show_info(name: &str) {
    println!("(info not yet implemented for '{}')", name);
}

fn load_file(file: &str) {
    println!("Loading {}...", file);
    // TODO: Load and compile the file
    println!("(file loading not yet implemented)");
}

fn set_option(opt: &str) {
    let parts: Vec<&str> = opt.split_whitespace().collect();
    match parts.as_slice() {
        ["profile", profile] => {
            println!("Setting profile to: {}", profile);
        }
        ["edition", edition] => {
            println!("Setting edition to: {}", edition);
        }
        _ => {
            println!("Unknown option: {}", opt);
        }
    }
}
