import React, { useState } from "react";
import VideoStreamer from "./components/VideoStreamer";
import CaptionReader from "./components/CaptionReader";

export default function App() {

  const [video, setVideo] = useState(true);

  return <main className="flex flex-col items-center p-5">
    <button 
    className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    onClick={() => {setVideo(!video);}}
    >
      Change Mode
    </button>
    <h1 className="text-2xl font-bold text-gray-800">Live Video Streamer</h1>
    {video ? <VideoStreamer></VideoStreamer>
    : <CaptionReader
      videoUrl={'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'}
    ></CaptionReader>}
  </main>
}