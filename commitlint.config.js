module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'scope-empty': [2, 'never'],
    'footer-max-line-length': [0, 'never'],
    'body-max-line-length': [0, 'never'],
  },
  ignores: [(commit) => commit.startsWith('revert:')]
};
