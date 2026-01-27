/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_URL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001'}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
