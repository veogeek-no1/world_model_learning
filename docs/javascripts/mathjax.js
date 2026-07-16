window.MathJax = {
  // tex-mml-chtml 是精简包，不含 boldsymbol；不显式加载的话
  // \boldsymbol{\epsilon} 会渲染成刺眼的红色报错字面量。
  loader: {
    load: ["[tex]/boldsymbol"],
  },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// navigation.instant 是 SPA 式换页，不会触发 MathJax 的初始渲染，
// 必须在每次换页后手动重排，否则跳转过去的页面公式是源码。
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
