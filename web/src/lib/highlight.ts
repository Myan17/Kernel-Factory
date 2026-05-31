// Tiny, dependency-free Python tokenizer. Good enough to make our generated
// kernels look great and — crucially — to treat {placeholders} as first-class
// tokens so the Assemble stage can morph them into integers.

export type TokenType =
  | "kw"
  | "builtin"
  | "str"
  | "num"
  | "comment"
  | "placeholder"
  | "def"
  | "deco"
  | "punct"
  | "plain";

export interface Token {
  text: string;
  type: TokenType;
  /** for placeholders: the bare key, e.g. "block_m" */
  key?: string;
}

const KEYWORDS = new Set([
  "import", "from", "as", "def", "return", "lambda", "for", "in", "if",
  "else", "elif", "while", "with", "class", "and", "or", "not", "is",
  "None", "True", "False", "yield", "pass",
]);

const BUILTINS = new Set([
  "jax", "jnp", "pl", "pltpu", "self", "shape", "astype", "zeros_like",
  "dot", "sqrt", "mean", "print", "range", "max", "len",
]);

const TOKEN_RE = new RegExp(
  [
    "(#[^\\n]*)", // 1 comment
    "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"[^\"\\n]*\"|'[^'\\n]*')", // 2 string
    "(\\{[a-zA-Z_][a-zA-Z0-9_]*\\})", // 3 placeholder
    "(@[A-Za-z_][A-Za-z0-9_]*)", // 4 decorator
    "(\\b\\d+\\.?\\d*(?:e-?\\d+)?\\b)", // 5 number
    "([A-Za-z_][A-Za-z0-9_]*)", // 6 word
    "(\\s+)", // 7 whitespace
    "([^\\s])", // 8 single other char
  ].join("|"),
  "g"
);

export function tokenizePython(code: string): Token[] {
  const tokens: Token[] = [];
  let m: RegExpExecArray | null;
  let prevWord = "";
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(code)) !== null) {
    const [, comment, str, placeholder, deco, num, word, ws, other] = m;
    if (comment !== undefined) tokens.push({ text: comment, type: "comment" });
    else if (str !== undefined) tokens.push({ text: str, type: "str" });
    else if (placeholder !== undefined)
      tokens.push({ text: placeholder, type: "placeholder", key: placeholder.slice(1, -1) });
    else if (deco !== undefined) tokens.push({ text: deco, type: "deco" });
    else if (num !== undefined) tokens.push({ text: num, type: "num" });
    else if (word !== undefined) {
      let type: TokenType = "plain";
      if (KEYWORDS.has(word)) type = "kw";
      else if (prevWord === "def" || prevWord === "class") type = "def";
      else if (BUILTINS.has(word)) type = "builtin";
      tokens.push({ text: word, type });
      prevWord = word;
      continue;
    } else if (ws !== undefined) tokens.push({ text: ws, type: "plain" });
    else if (other !== undefined) tokens.push({ text: other, type: "punct" });
    prevWord = "";
  }
  return tokens;
}

// Fixed, bright-on-dark syntax colors. Code blocks always sit on a dark
// background (bg-codebg) regardless of theme, so these must NOT depend on the
// theme-inverting --ink/--muted vars.
export const TOKEN_CLASS: Record<TokenType, string> = {
  kw: "text-[#9d9bff] font-medium",
  builtin: "text-[#c4a7fb]",
  str: "text-[#5fe3a1]",
  num: "text-[#ffce5c]",
  comment: "text-white/40 italic",
  placeholder: "",
  def: "text-[#ff9b8b] font-medium",
  deco: "text-[#ff9b8b]",
  punct: "text-white/55",
  plain: "text-white/85",
};
