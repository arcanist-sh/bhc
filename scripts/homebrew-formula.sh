#!/bin/sh
# Print the Homebrew formula for a bhc release (arcanist-sh/homebrew-tap,
# Formula/bhc.rb).
#
# Usage: scripts/homebrew-formula.sh <version> <SHA256SUMS.txt>
#   <version>  the release tag, with or without the leading "v" (v0.2.29)
#   <file>     the release's SHA256SUMS.txt
#
# The release workflow runs this to update the tap after every release; it can
# also be run by hand:
#
#   gh release download v0.2.29 -R arcanist-sh/bhc -p SHA256SUMS.txt -O /tmp/sums
#   scripts/homebrew-formula.sh v0.2.29 /tmp/sums > ../homebrew-tap/Formula/bhc.rb

set -eu

if [ $# -ne 2 ]; then
    echo "usage: $0 <version> <SHA256SUMS.txt>" >&2
    exit 2
fi

version="${1#v}"
sums="$2"

# The checksum of an archive. Older releases list it as
# "<dir>/<archive>", newer ones as the bare "<archive>".
sha_of() {
    sha=$(awk -v f="$1" '{ n = $2; sub(/^.*\//, "", n); if (n == f) print $1 }' "$sums")
    if [ -z "$sha" ]; then
        echo "error: no checksum for $1 in $sums" >&2
        exit 1
    fi
    echo "$sha"
}

darwin_sha=$(sha_of bhc-aarch64-darwin.tar.gz)
linux_sha=$(sha_of bhc-x86_64-linux.tar.gz)
base="https://github.com/arcanist-sh/bhc/releases/download/v${version}"

cat <<EOF
class Bhc < Formula
  desc "Basel Haskell Compiler - native code Haskell compiler"
  homepage "https://arcanist.sh/bhc/"
  version "${version}"
  license "BSD-3-Clause"

  on_macos do
    if Hardware::CPU.arm?
      url "${base}/bhc-aarch64-darwin.tar.gz"
      sha256 "${darwin_sha}"
    else
      odie "bhc is not available for Intel Macs. Apple Silicon required."
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "${base}/bhc-x86_64-linux.tar.gz"
      sha256 "${linux_sha}"
    else
      odie "bhc is not available for Linux ARM. x86_64 required."
    end
  end

  def install
    bin.install "bhc"
    # The REPL (\`bhc repl\` runs the bhci next to bhc) and the IR inspector,
    # in releases that ship them.
    %w[bhci bhi].each { |tool| bin.install tool if File.exist?(tool) }
    # The driver links the RTS and the stdlib archives, not only the RTS.
    lib.install Dir["lib/*.a"]
    pkgshare.install "examples" if File.directory?("examples")
  end

  def caveats
    <<~EOS
      To compile Haskell programs, bhc needs to find its runtime libraries.
      Add to your shell profile:

        export LIBRARY_PATH="#{opt_lib}:\$LIBRARY_PATH"

      Example usage:

        echo 'main = putStrLn "Hello!"' > hello.hs
        bhc hello.hs -o hello
        ./hello
    EOS
  end

  test do
    assert_match "bhc", shell_output("#{bin}/bhc --version")
    (testpath/"hello.hs").write 'main = putStrLn "Hello!"'
    ENV["LIBRARY_PATH"] = lib.to_s
    system bin/"bhc", "hello.hs", "-o", "hello"
    assert_equal "Hello!\\n", shell_output("./hello")
  end
end
EOF
