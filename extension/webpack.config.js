// ====================================================
// Webpack Configuration for Emotion Lens Extension
// Bundles TypeScript, React, CSS into Chrome Extension
// ====================================================

const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const isProd = argv.mode === 'production';

  return {
    context: __dirname,  // Always resolve from extension directory
    entry: {
      background: './src/background/background.ts',
      content: './src/content/contentScript.ts',
      popup: './src/popup/popup.tsx',
      options: './src/options/options.tsx',
      sidepanel: './src/sidepanel/sidepanel.tsx',
    },
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: '[name].js',
      // Chrome blocks filenames starting with '_' - use safe prefix
      chunkFilename: 'chunk.[id].[contenthash:8].js',
      clean: true,
    },
    resolve: {
      extensions: ['.ts', '.tsx', '.js', '.jsx'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    module: {
      rules: [
        {
          test: /\.tsx?$/,
          use: {
            loader: 'ts-loader',
            options: {
              transpileOnly: true,
              compilerOptions: {
                jsx: 'react-jsx',
              },
            },
          },
          exclude: /node_modules/,
        },
        {
          test: /\.css$/,
          use: [
            isProd ? MiniCssExtractPlugin.loader : 'style-loader',
            'css-loader',
          ],
        },
        {
          test: /\.(png|svg|jpg|jpeg|gif)$/i,
          type: 'asset/resource',
        },
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/i,
          type: 'asset/resource',
        },
      ],
    },
    // Disable code splitting to avoid Chrome's _-prefixed filename restriction
    optimization: {
      splitChunks: false,
      chunkIds: 'named',
    },
    plugins: [
      // Copy static assets
      new CopyPlugin({
        patterns: [
          {
            from: 'manifest.json',
            to: 'manifest.json',
            context: __dirname,
          },
          {
            from: 'public/icons',
            to: 'icons',
            noErrorOnMissing: true,
          },
          {
            from: 'public/wasm',
            to: 'wasm',
            noErrorOnMissing: true,
          },
        ],
      }),

      // HTML pages for UI components
      new HtmlWebpackPlugin({
        template: './src/popup/popup.html',
        filename: 'popup/index.html',
        chunks: ['popup'],
        inject: true,
      }),
      new HtmlWebpackPlugin({
        template: './src/options/options.html',
        filename: 'options/index.html',
        chunks: ['options'],
        inject: true,
      }),
      new HtmlWebpackPlugin({
        template: './src/sidepanel/sidepanel.html',
        filename: 'sidepanel/index.html',
        chunks: ['sidepanel'],
        inject: true,
      }),

      // Extract CSS in production
      ...(isProd
        ? [
            new MiniCssExtractPlugin({
              filename: '[name].css',
            }),
          ]
        : []),
    ],
    devtool: isProd ? false : 'inline-source-map',
    stats: {
      colors: true,
      assets: true,
      modules: false,
      chunks: false,
    },
  };
};