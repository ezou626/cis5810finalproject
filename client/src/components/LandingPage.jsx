import React from 'react';

const LandingPage = ({ onGetStarted }) => {
  return (
    <div className="bg-gray-900 text-white min-h-screen">
      {/* Hero Section */}
  <header className="relative">
  <div className="max-w-7xl mx-auto px-6 py-16 text-center">
    <div className="flex items-center justify-center space-x-4">
      {/* Stephen A. Smith Image */}
      <img 
        src="/stephen.png" 
        alt="Stephen A. Smith" 
        className="w-28 h-24 rounded-full shadow-lg"
      />
      {/* Title */}
      <h1 className="text-5xl md:text-6xl font-extrabold leading-tight">
        Stephen AI Smith
      </h1>
    </div>
    <p className="mt-6 text-xl md:text-2xl text-gray-300">
      The Ultimate AI-Powered Video Commentator and Auto-Zooming Experience. 
      <br /> Dive into Precision Tracking and Real-Time Commentary.
    </p>
    <div className="mt-8 flex justify-center space-x-4">
      <a 
        href="#features" 
        onClick={onGetStarted}
        className="bg-indigo-600 hover:bg-indigo-700 px-8 py-4 text-lg font-bold rounded-md shadow-md"
      >
        Explore Features
      </a>
      <button 
        onClick={onGetStarted}
        className="bg-green-600 hover:bg-green-700 px-8 py-4 text-lg font-bold rounded-md shadow-md"
      >
        Get Started
      </button>
    </div>
  </div>
</header>

      {/* Features Section */}
      <section id="features" className="py-24 bg-gray-800">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-12">Why Use Stephen AI Smith?</h2>
          <div className="grid md:grid-cols-3 gap-12">
            <Feature 
              icon="🚀"
              title="Auto-Zoom Tracking"
              description="AI-driven object tracking with automatic zoom and smooth transitions. Stay locked on your subject with precision control."
            />
            <Feature 
              icon="💬"
              title="Real-Time Commentary"
              description="AI-powered real-time commentary for videos. Watch, learn, and listen to intelligent insights as events unfold."
            />
            <Feature 
              icon="🛠️"
              title="Customizable Settings"
              description="Easily adjust tracking, zoom speed, and commentary modes for a tailored experience. Full control at your fingertips."
            />
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 bg-gray-800">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-12">What Our Users Say</h2>
          <div className="grid md:grid-cols-3 gap-12">
            <Testimonial 
              image="https://randomuser.me/api/portraits/men/32.jpg"
              name="Alex Johnson"
              quote="This app revolutionized my video workflow. The zoom tracking is smooth, and the AI commentary adds a whole new dimension."
            />
            <Testimonial 
              image="https://randomuser.me/api/portraits/women/44.jpg"
              name="Sarah Lee"
              quote="I can't believe how accurate the auto-tracking is. I use it for sports videos, and it works like a charm!"
            />
            <Testimonial 
              image="https://randomuser.me/api/portraits/men/65.jpg"
              name="Michael Smith"
              quote="ZoomBot AI has made my video production 10x faster. The ability to zoom automatically and hear live commentary is a game changer."
            />
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 bg-indigo-600">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold mb-6 text-white">Get Started with Stephen AI Smith Today</h2>
          <a 
            href="#"
            className="bg-gray-900 hover:bg-gray-800 px-8 py-4 text-lg font-bold rounded-md shadow-md text-white"
          >
            Get Early Access
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-400">
            &copy; 2024 Stephen AI Smith. All Rights Reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

// Reusable Feature Component
const Feature = ({ icon, title, description }) => (
  <div className="flex flex-col items-center text-center">
    <div className="text-6xl mb-4">{icon}</div>
    <h3 className="text-2xl font-bold mb-4">{title}</h3>
    <p className="text-gray-300">{description}</p>
  </div>
);

// Reusable Testimonial Component
const Testimonial = ({ image, name, quote }) => (
  <div className="bg-gray-900 p-8 rounded-lg shadow-lg text-center">
    <img 
      src={image} 
      alt={name} 
      className="w-24 h-24 rounded-full mx-auto mb-4"
    />
    <p className="italic text-gray-300 mb-4">"{quote}"</p>
    <h4 className="text-xl font-bold">{name}</h4>
  </div>
);

export default LandingPage;
