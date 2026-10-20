import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import SplashScreen from './components/SplashScreen';
import LoginScreen from './components/LoginScreen';
import SignUpScreen from './components/SignUpScreen';
import PatientDashboard from './components/PatientDashboard';
import DoctorDashboard from './components/DoctorDashboard';
import AdminDashboard from './components/AdminDashboard';
import { getSession } from './utils/api';
import { Toaster } from 'react-hot-toast';
import { MessageSquare, X } from 'lucide-react';
import Chatbot from './components/Chatbot';
import './index.css';

function ProtectedRoute({ children, allowedRoles }) {
  const session = getSession();
  if (!session) return <Navigate to="/login" replace />;
  if (allowedRoles && !allowedRoles.includes(session.user.role)) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function RoleRedirect() {
  const session = getSession();
  if (!session) return <Navigate to="/login" replace />;
  const { role } = session.user;
  if (role === 'Patient') return <Navigate to="/patient" replace />;
  if (role === 'Doctor') return <Navigate to="/doctor" replace />;
  if (role === 'Admin') return <Navigate to="/admin" replace />;
  return <Navigate to="/login" replace />;
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [isChatOpen, setIsChatOpen] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 2800);
    return () => clearTimeout(timer);
  }, []);

  if (showSplash) return <SplashScreen />;

  return (
    <>
      <Toaster position="top-center" toastOptions={{ className: 'font-medium text-sm rounded-xl shadow-lg' }} />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<RoleRedirect />} />
          <Route path="/login" element={<LoginScreen />} />
          <Route path="/signup" element={<SignUpScreen />} />
          <Route path="/patient/*" element={
            <ProtectedRoute allowedRoles={['Patient']}>
              <PatientDashboard />
            </ProtectedRoute>
          } />
          <Route path="/doctor/*" element={
            <ProtectedRoute allowedRoles={['Doctor']}>
              <DoctorDashboard />
            </ProtectedRoute>
          } />
          <Route path="/admin/*" element={
            <ProtectedRoute allowedRoles={['Admin']}>
              <AdminDashboard />
            </ProtectedRoute>
          } />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>

        {/* Global Floating Multilingual AI Assistant */}
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
          {isChatOpen && (
            <div className="mb-3 animate-slide-up shadow-2xl rounded-2xl overflow-hidden border border-slate-200">
              <Chatbot isEmbedded={false} />
            </div>
          )}
          <button
            onClick={() => setIsChatOpen(!isChatOpen)}
            className="flex items-center gap-2.5 bg-gradient-to-r from-teal-600 via-teal-700 to-cyan-800 hover:from-teal-700 hover:to-cyan-900 text-white px-4 py-3 rounded-full shadow-xl hover:shadow-2xl transition-all duration-200 font-semibold text-sm group scale-100 hover:scale-105 active:scale-95 border border-white/20"
            title="Ask OsteoCare AI Assistant"
          >
            {isChatOpen ? <X size={20} /> : <MessageSquare size={20} className="group-hover:rotate-12 transition-transform" />}
            <span className="hidden sm:inline font-medium">
              {isChatOpen ? "Close AI Assistant" : "Ask AI (English / हिन्दी / ਪੰਜਾਬੀ)"}
            </span>
          </button>
        </div>
      </BrowserRouter>
    </>
  );
}
