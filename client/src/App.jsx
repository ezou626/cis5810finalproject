import React from "react";
import VideoStreamer from "./components/VideoStreamer";

export default function App() {
  return <main className="flex flex-col items-center p-5">
    <h1 className="text-2xl font-bold text-gray-800">Live Video Streamer</h1>
    <VideoStreamer></VideoStreamer>
  </main>
}