#lang racket
(for ([path (in-directory)]
     #:when (regexp-match? #rx"[.].rkt$" path))
  (printf "source file: ~a\n" path))
