import React, { useEffect, useState } from "react";

const backendUrl = import.meta.env.VITE_BACKEND_URL;

const CaptionReader = ({ videoUrl }) => {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    // Handle SSE connection
    const eventSource = new EventSource(
      `${backendUrl}/captions?video_url=${encodeURIComponent(videoUrl)}`);
    // const eventSource = new EventSource(
    //   `${backendUrl}/captions_debug`);

    eventSource.onopen = () => {
      console.log('EventSource connected')
    }

    eventSource.onmessage = (event) => {
      const newMessage = event.data;
      console.log(newMessage);
      setMessage(newMessage);
    };

    eventSource.onerror = (event) => {
      setError("Error connecting to the stream.");
      console.log(event);
      eventSource.close();
    };

    // Cleanup when the component unmounts
    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <div className="flex flex-col items-center p-4 space-y-6">
      <h1 className="text-xl font-semibold text-gray-900">Server-Sent Event Stream</h1>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="w-full max-w-md">
        <h2 className="text-lg font-medium text-gray-700">Last Message:</h2>
        <ul className="mt-4 space-y-2">
          {message.length > 0 ? <p className="text-gray-500">{message}</p> : (
            <p className="text-gray-500">Waiting for new messages...</p>
          )}
        </ul>
      </div>
    </div>
  );
};

export default CaptionReader;