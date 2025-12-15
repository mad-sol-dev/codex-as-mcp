# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2025.10.20.1] - 2025-10-20

### 📚 Documentation
- Align README, README.zh-CN, and AGENTS tool docs with current `spawn_agent`/`spawn_agents_parallel` behavior.

### 🔢 Versioning
- Release `2025.10.20.1` and publish to PyPI.

## [2025.9.16] - 2025-09-16

### 🛠️ Fixed
- Replace console emojis with ASCII labels to prevent UnicodeEncodeError on Windows (GBK) terminals.

## [2025.9.4] - 2025-09-03

### 🛠️ Fixed
- **Windows Compatibility**: Fixed `[WinError 2] 系统找不到指定的文件` (system cannot find file) error when launching the Codex CLI from MCP tools.
- Added cross-platform executable resolution using `shutil.which()`.
- Enhanced error handling with `FileNotFoundError` catching across all MCP tools.
- Improved error messages with Windows-specific guidance and installation instructions.

### 🔧 Changed  
- Added platform detection to provide Windows-specific error messages
- Enhanced subprocess command resolution for cross-platform compatibility
- Updated error handling to guide Windows users to proper installation steps or WSL usage

### 📚 Documentation
- Windows users now receive helpful error messages about experimental support
- Added guidance for npm installation and PATH configuration
- Included WSL recommendation for better Windows compatibility

## [0.1.16] - 2025-08-28

### 🛠️ Fixed
- **BREAKING**: Fixed "list index out of range" error while parsing Codex CLI output from MCP tools.
- Added comprehensive defensive checks for empty Codex output blocks
- Enhanced error handling with detailed diagnostic information
- Improved subprocess handling to capture output even on command failures

### 🔧 Changed
- **BREAKING**: Updated command structure to highlight safe mode and YOLO-style writable mode with explicit flags.
- Adjusted docs and defaults for Codex CLI version >= 0.25.0.

### 📚 Documentation
- Added prominent version requirement warnings in README files
- Updated installation instructions to use `@latest` tag
- Added version verification steps
- Emphasized compatibility requirements

### 🧪 Internal
- Enhanced error messages with command details and output previews
- Added explicit IndexError handling alongside existing ValueError handling
- Improved CalledProcessError handling with captured output

## [0.1.15] - Previous Release

### Features
- Safe mode implementation with read-only sandbox
- Writable mode with --yolo flag
- Sequential execution to prevent conflicts
- Initial MCP tool surface for launching Codex agents

---

## Migration Guide for v0.1.16

### Breaking Changes

1. **Codex CLI Version**: Update to version 0.25.0 or later:
   ```bash
   npm install -g @openai/codex@latest
   codex --version  # Verify >= 0.25.0
   ```

2. **Command Flags**: The server drives the Codex CLI with the standard `--full-auto` and `--skip-git-repo-check` flags. If you rely on custom wrappers or stricter sandboxing, update them accordingly.

3. **Error Handling**: Improved error messages may look different but provide more diagnostic information

### What Stays the Same

- MCP server configuration in `.mcp.json` remains unchanged
- Tool signatures (`spawn_agent` and `spawn_agents_parallel`) remain the same
- Safe mode vs YOLO mode behavior is unchanged
- All documented features work the same way

### Benefits

- ✅ Eliminates "list index out of range" crashes
- ✅ Better error diagnostics for troubleshooting
- ✅ More robust command execution
- ✅ Compatible with latest Codex CLI features
