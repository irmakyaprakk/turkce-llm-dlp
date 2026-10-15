"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {maskDetections, pythonOffsetToJsIndex, restoreText} = require("./core.js");

test("converts Python code point offsets to JavaScript UTF-16 indexes", () => {
  const text = "😀 TCKN 10000000146";
  const pythonStart = [...text].indexOf("1");
  assert.equal(pythonOffsetToJsIndex(text, pythonStart), text.indexOf("1"));
});

test("masks repeat values consistently and restores locally", () => {
  const text = "TCKN 10000000146 / TCKN 100-000-00146";
  const parts = [...text];
  const spans = [];
  for (const value of ["10000000146", "100-000-00146"]) {
    const start = parts.join("").indexOf(value);
    spans.push({type: "TCKN", start: [...text.slice(0, start)].length, end: [...text.slice(0, start + value.length)].length});
  }
  const result = maskDetections(text, spans, "abcd1234");
  assert.equal(result.values.size, 1);
  assert.equal((result.masked.match(/<TCKN_1_abcd1234>/g) || []).length, 2);
  assert.equal(restoreText(result.masked, result.values), "TCKN 10000000146 / TCKN 10000000146");
});

test("rejects overlapping or out-of-range offsets", () => {
  assert.throws(() => maskDetections("abcdef", [
    {type: "PERSON", start: 0, end: 4}, {type: "ORG", start: 3, end: 6}
  ], "abcd1234"), /overlap/);
  assert.throws(() => maskDetections("abc", [{type: "PERSON", start: 0, end: 9}], "abcd1234"), /exceeds/);
});
