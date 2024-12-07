import React, { useEffect, useState } from "react";

const backendUrl = import.meta.env.VITE_BACKEND_URL;

const CaptionReader = ({ streamUrl }) => {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!streamUrl) return;
    // Handle SSE connection
    const eventSource = new EventSource(
      `${backendUrl}/captions?video_url=${encodeURIComponent(streamUrl)}`);
    // const eventSource = new EventSource(
    //   `${backendUrl}/captions_debug`);

    eventSource.onopen = () => {
      console.log('EventSource connected')
    }

    eventSource.onmessage = (event) => {
      const newMessage = event.data;
      console.log(newMessage);
      setError("");
      setMessage((prevMessage) =>newMessage);
    };

    eventSource.onerror = (event) => {
      setError("");
      console.log(event);
      eventSource.close();
    };

    // Cleanup when the component unmounts
    return () => {
      eventSource.close();
    };
  }, [streamUrl]);

  return (
    <div className="flex flex-col items-center w-1/2 md:w-1/4 bg-white rounded-lg shadow-lg p-6 space-y-6">
      {streamUrl ? <>
      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="w-full w-md">
        <h2 className="text-lg font-medium text-gray-700 w-md">Last Message:</h2>
        <ul className="mt-4 space-y-2">
          {message.length > 0 ? <p className="text-gray-500 w-md">{message}</p> : (
            <p className="text-gray-500">Waiting for new messages...</p>
          )}
        </ul>
      </div></> : (
        <p className="text-gray-500">No live stream is currently active.</p>
      )
      }
    </div>
  );
};

export default CaptionReader;