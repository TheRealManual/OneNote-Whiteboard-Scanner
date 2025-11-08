const path = require('path');

module.exports = {
  mode: 'development',
  entry: './renderer/index.jsx',
  output: {
    path: path.resolve(__dirname, 'renderer'),
    filename: 'app.bundle.js'
  },
  target: 'electron-renderer',
  module: {
    rules: [
      {
        test: /\.jsx?$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-react']
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  resolve: {
    extensions: ['.js', '.jsx']
  }
};
