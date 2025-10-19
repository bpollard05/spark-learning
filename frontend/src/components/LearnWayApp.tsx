'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, Book, MessageSquare, Brain, Sparkles, Home, BarChart3, Zap, Clock, Target, BookOpen, Lightbulb, PenTool, CheckCircle2, Video, FileText, Calendar, FolderOpen, Plus, ChevronRight, TrendingUp, Award, Flame, Star } from 'lucide-react';

// DEMO DATA - 4th Grade Student
const DEMO_CLASSES = [
  { id: '1', name: 'Math', color: 'bg-blue-500', progress: 78, nextTopic: 'Multiplication & Division', icon: '🔢' },
  { id: '2', name: 'Reading & Writing', color: 'bg-purple-500', progress: 85, nextTopic: 'Charlotte\'s Web', icon: '📚' },
  { id: '3', name: 'Science', color: 'bg-green-500', progress: 72, nextTopic: 'Solar System', icon: '🔬' },
  { id: '4', name: 'Social Studies', color: 'bg-orange-500', progress: 80, nextTopic: 'U.S. States & Geography', icon: '🌎' },
];

const DEMO_NOTES = [
  { id: '1', class: 'Math', title: 'Multiplication Tables', date: '2 days ago', preview: 'Practice: 7 × 8 = 56, 9 × 6 = 54...' },
  { id: '2', class: 'Reading & Writing', title: 'Story Elements', date: '1 week ago', preview: 'Characters, Setting, Plot, Problem, Solution...' },
  { id: '3', class: 'Science', title: 'The Water Cycle', date: '3 days ago', preview: 'Evaporation → Condensation → Precipitation...' },
  { id: '4', class: 'Social Studies', title: 'Map Reading Skills', date: '5 days ago', preview: 'Compass rose, map key, scale...' },
];

const DEMO_CALENDAR = [
  { date: 'Oct 20', event: 'Math Quiz - Times Tables', type: 'quiz', class: 'Math' },
  { date: 'Oct 22', event: 'Charlotte\'s Web Chapter 5', type: 'reading', class: 'Reading & Writing' },
  { date: 'Oct 24', event: 'Science Project: Planets', type: 'project', class: 'Science' },
  { date: 'Oct 25', event: 'States & Capitals Test', type: 'quiz', class: 'Social Studies' },
  { date: 'Oct 27', event: 'Book Report Due', type: 'assignment', class: 'Reading & Writing' },
];

const DEMO_RECENT_ACTIVITY = [
  { action: 'Practiced division problems', class: 'Math', time: '2 hours ago', icon: CheckCircle2, color: 'text-green-500' },
  { action: 'Read Chapter 4 of Charlotte\'s Web', class: 'Reading & Writing', time: '5 hours ago', icon: BookOpen, color: 'text-purple-500' },
  { action: 'Learned about planets', class: 'Science', time: '1 day ago', icon: Brain, color: 'text-green-500' },
  { action: 'Practiced U.S. states quiz', class: 'Social Studies', time: '2 days ago', icon: Star, color: 'text-orange-500' },
];

const DEMO_BOOKS = [
  { id: '1', title: 'Charlotte\'s Web', author: 'E.B. White', progress: 62, class: 'Reading & Writing', cover: 'bg-pink-400', emoji: '🕷️' },
  { id: '2', title: 'The Magic School Bus: Solar System', author: 'Joanna Cole', progress: 45, class: 'Science', cover: 'bg-blue-400', emoji: '🚌' },
  { id: '3', title: 'If You Made a Million', author: 'David M. Schwartz', progress: 80, class: 'Math', cover: 'bg-green-400', emoji: '💰' },
];

const DEMO_ACHIEVEMENTS = [
  { title: 'Math Master', description: 'Completed 50 practice problems', icon: '🏆', color: 'bg-yellow-100' },
  { title: 'Reading Star', description: 'Read for 5 days in a row', icon: '⭐', color: 'bg-purple-100' },
  { title: 'Science Explorer', description: 'Finished Water Cycle project', icon: '🔬', color: 'bg-green-100' },
];

export default function LearnWayApp() {
  const [user, setUser] = useState<any>(null);
  const [currentView, setCurrentView] = useState('home');
  const [currentSession, setCurrentSession] = useState<any>(null);
  const [selectedClass, setSelectedClass] = useState<any>(null);

  useEffect(() => {
    const demoUser = {
      id: 'demo-user',
      email: 'alex.student@school.edu',
      display_name: 'Alex',
      role: 'student',
      grade: '4th Grade',
      school: 'Sunshine Elementary School',
      teacher: 'Mrs. Rodriguez'
    };
    setUser(demoUser);
  }, []);

  if (!user) {
    return <LoginView onLogin={setUser} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} user={user} />
      <div className="ml-64 p-8">
        {currentView === 'home' && <StudentHub user={user} onStartSession={(s: any) => { setCurrentSession(s); setCurrentView('learn'); }} />}
        {currentView === 'learn' && <LearnMode session={currentSession} />}
        {currentView === 'classes' && <ClassesView onSelectClass={(c: any) => { setSelectedClass(c); setCurrentView('classDetail'); }} />}
        {currentView === 'classDetail' && <ClassDetailView classData={selectedClass} onBack={() => setCurrentView('classes')} />}
        {currentView === 'notes' && <NotesView />}
        {currentView === 'calendar' && <CalendarView />}
        {currentView === 'read' && <ReadingMode />}
        {currentView === 'analytics' && <AnalyticsView />}
      </div>
    </div>
  );
}

function Sidebar({ currentView, setCurrentView, user }: any) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'classes', label: 'My Classes', icon: BookOpen },
    { id: 'notes', label: 'My Notes', icon: FileText },
    { id: 'calendar', label: 'Calendar', icon: Calendar },
    { id: 'read', label: 'Reading', icon: Book },
    { id: 'analytics', label: 'My Progress', icon: TrendingUp },
  ];

  return (
    <div className="fixed left-0 top-0 h-screen w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <Sparkles className="w-8 h-8 text-indigo-600" />
          <h1 className="text-2xl font-bold text-gray-900">Spark Learning</h1>
        </div>
        <p className="text-sm text-gray-500 mt-1">Learn Your Way! ✨</p>
      </div>

      <nav className="flex-1 p-4">
        {navItems.map(item => (
          <button
            key={item.id}
            onClick={() => setCurrentView(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-colors ${
              currentView === item.id || (currentView === 'classDetail' && item.id === 'classes')
                ? 'bg-indigo-50 text-indigo-600'
                : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
            <span className="text-white font-semibold text-lg">A</span>
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900">{user.display_name}</p>
            <p className="text-xs text-gray-500">{user.grade}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function LoginView({ onLogin }: any) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
        <div className="flex items-center justify-center gap-2 mb-8">
          <Sparkles className="w-12 h-12 text-indigo-600" />
          <h1 className="text-3xl font-bold text-gray-900">Spark Learning</h1>
        </div>
        <div className="text-center">
          <p className="text-xl text-gray-700 mb-2 font-semibold">Welcome Back! 🎉</p>
          <p className="text-gray-600 mb-6">Your AI learning friend is ready to help!</p>
          <button
            onClick={() => onLogin({ display_name: 'Alex', role: 'student' })}
            className="w-full bg-indigo-600 text-white py-4 rounded-lg font-medium hover:bg-indigo-700 transition-colors text-lg"
          >
            Start Learning! 🚀
          </button>
        </div>
      </div>
    </div>
  );
}

function StudentHub({ user, onStartSession }: any) {
  const [showConfig, setShowConfig] = useState(false);
  const [sessionConfig, setSessionConfig] = useState({
    goal: 'understand',
    energy: 'medium',
    duration: 30,
    inputPreference: 'mixed',
  });

  const startSession = () => {
    onStartSession({ id: 'demo-session', ...sessionConfig });
  };

  if (showConfig) {
    return (
      <div className="max-w-2xl mx-auto">
        <button onClick={() => setShowConfig(false)} className="text-indigo-600 hover:text-indigo-700 mb-6 font-medium">
          ← Back to Home
        </button>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Let's Start Learning! 🎯</h2>
          <p className="text-gray-600 mb-6">Tell me what you want to learn today</p>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                What do you want to do?
              </label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { value: 'understand', label: 'Learn Something New', icon: Brain, emoji: '🧠' },
                  { value: 'practice', label: 'Practice Problems', icon: PenTool, emoji: '✏️' },
                  { value: 'review', label: 'Quick Review', icon: Clock, emoji: '⚡' },
                  { value: 'explore', label: 'Explore & Discover', icon: Lightbulb, emoji: '💡' },
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => setSessionConfig({...sessionConfig, goal: option.value})}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      sessionConfig.goal === option.value
                        ? 'border-indigo-600 bg-indigo-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-3xl mb-2">{option.emoji}</div>
                    <p className="text-sm font-medium text-gray-900">{option.label}</p>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                How are you feeling?
              </label>
              <div className="flex gap-3">
                {[
                  { value: 'low', label: 'Tired', emoji: '😴' },
                  { value: 'medium', label: 'Good', emoji: '😊' },
                  { value: 'high', label: 'Energetic!', emoji: '🚀' }
                ].map(level => (
                  <button
                    key={level.value}
                    onClick={() => setSessionConfig({...sessionConfig, energy: level.value})}
                    className={`flex-1 py-3 rounded-lg border-2 transition-all ${
                      sessionConfig.energy === level.value
                        ? 'border-indigo-600 bg-indigo-50 text-indigo-600'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-2xl mb-1">{level.emoji}</div>
                    <div className="text-sm font-medium">{level.label}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                How long do you want to study?
              </label>
              <input
                type="range"
                min="15"
                max="60"
                step="15"
                value={sessionConfig.duration}
                onChange={(e) => setSessionConfig({...sessionConfig, duration: parseInt(e.target.value)})}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-500 mt-2">
                <span>15 min</span>
                <span className="font-medium text-indigo-600 text-lg">{sessionConfig.duration} minutes</span>
                <span>60 min</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                How do you like to learn?
              </label>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { value: 'reading', label: 'Reading', icon: BookOpen, emoji: '📖' },
                  { value: 'visual', label: 'Pictures & Videos', icon: Video, emoji: '🎨' },
                  { value: 'mixed', label: 'Mix of Both', icon: Sparkles, emoji: '✨' },
                ].map(option => (
                  <button
                    key={option.value}
                    onClick={() => setSessionConfig({...sessionConfig, inputPreference: option.value})}
                    className={`p-3 rounded-lg border-2 transition-all ${
                      sessionConfig.inputPreference === option.value
                        ? 'border-indigo-600 bg-indigo-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-2xl mb-1">{option.emoji}</div>
                    <p className="text-xs font-medium text-gray-900">{option.label}</p>
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={startSession}
              className="w-full bg-indigo-600 text-white py-4 rounded-lg font-medium hover:bg-indigo-700 transition-colors text-lg"
            >
              Let's Go! 🚀
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Hi {user.display_name}! 👋
        </h1>
        <p className="text-gray-600">Welcome back to {user.school}!</p>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Flame className="w-5 h-5 text-orange-500" />
            <p className="text-sm font-medium text-gray-600">Learning Streak</p>
          </div>
          <p className="text-3xl font-bold text-gray-900">12 days 🔥</p>
          <p className="text-sm text-green-600 mt-1">Amazing work!</p>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Star className="w-5 h-5 text-yellow-500" />
            <p className="text-sm font-medium text-gray-600">Stars Earned</p>
          </div>
          <p className="text-3xl font-bold text-gray-900">156 ⭐</p>
          <p className="text-sm text-gray-500 mt-1">This month</p>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Award className="w-5 h-5 text-purple-500" />
            <p className="text-sm font-medium text-gray-600">Achievements</p>
          </div>
          <p className="text-3xl font-bold text-gray-900">8 🏆</p>
          <p className="text-sm text-gray-500 mt-1">Keep going!</p>
        </div>
      </div>

      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl shadow-xl p-8 mb-8 text-white">
        <h2 className="text-2xl font-bold mb-3">Ready to Learn? 🎓</h2>
        <p className="text-indigo-100 mb-6">
          Your AI learning buddy can help you with homework, practice problems, reading, and more!
        </p>
        <button
          onClick={() => setShowConfig(true)}
          className="bg-white text-indigo-600 px-6 py-3 rounded-lg font-medium hover:bg-indigo-50 transition-colors inline-flex items-center gap-2"
        >
          <Brain className="w-5 h-5" />
          Start Learning Session
        </button>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            📚 My Classes
          </h3>
          <div className="space-y-3">
            {DEMO_CLASSES.map(cls => (
              <div key={cls.id} className="p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 flex items-center gap-2">
                    <span className="text-xl">{cls.icon}</span>
                    {cls.name}
                  </span>
                  <span className="text-sm text-gray-500">{cls.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className={`${cls.color} h-2 rounded-full`} style={{ width: `${cls.progress}%` }} />
                </div>
                <p className="text-xs text-gray-500 mt-2">Next: {cls.nextTopic}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            📖 Currently Reading
          </h3>
          <div className="space-y-3">
            {DEMO_BOOKS.map(book => (
              <div key={book.id} className="p-3 rounded-lg bg-gray-50">
                <div className="flex items-start gap-3">
                  <div className={`w-12 h-16 ${book.cover} rounded flex items-center justify-center text-2xl`}>
                    {book.emoji}
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 text-sm">{book.title}</p>
                    <p className="text-xs text-gray-500 mb-2">{book.author}</p>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div className="bg-indigo-500 h-1.5 rounded-full" style={{ width: `${book.progress}%` }} />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{book.progress}% complete</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            ⏰ Coming Up
          </h3>
          <div className="space-y-3">
            {DEMO_CALENDAR.slice(0, 4).map((item, i) => (
              <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-gray-50">
                <div className="text-center">
                  <div className="text-xs font-medium text-gray-500">{item.date.split(' ')[0]}</div>
                  <div className="text-lg font-bold text-gray-900">{item.date.split(' ')[1]}</div>
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 text-sm">{item.event}</p>
                  <p className="text-xs text-gray-500">{item.class}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            🏆 Recent Achievements
          </h3>
          <div className="space-y-3">
            {DEMO_ACHIEVEMENTS.map((achievement, i) => (
              <div key={i} className={`p-4 rounded-lg ${achievement.color}`}>
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{achievement.icon}</span>
                  <div>
                    <p className="font-medium text-gray-900">{achievement.title}</p>
                    <p className="text-sm text-gray-600">{achievement.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ClassesView({ onSelectClass }: any) {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Classes 📚</h1>
        <p className="text-gray-600">Click on a class to see your work and progress</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {DEMO_CLASSES.map(cls => (
          <button
            key={cls.id}
            onClick={() => onSelectClass(cls)}
            className="bg-white rounded-xl shadow p-6 hover:shadow-lg transition-all text-left"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <span className="text-4xl">{cls.icon}</span>
                <div>
                  <h3 className="text-xl font-bold text-gray-900">{cls.name}</h3>
                  <p className="text-sm text-gray-500">4th Grade</p>
                </div>
              </div>
              <ChevronRight className="w-6 h-6 text-gray-400" />
            </div>
            
            <div className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Progress</span>
                <span className="font-medium text-gray-900">{cls.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className={`${cls.color} h-2 rounded-full`} style={{ width: `${cls.progress}%` }} />
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Coming up next:</p>
              <p className="text-sm font-medium text-gray-900">{cls.nextTopic}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ClassDetailView({ classData, onBack }: any) {
  const classNotes = DEMO_NOTES.filter(note => note.class === classData.name);
  const classEvents = DEMO_CALENDAR.filter(event => event.class === classData.name);

  return (
    <div className="max-w-6xl mx-auto">
      <button onClick={onBack} className="text-indigo-600 hover:text-indigo-700 mb-6 font-medium">
        ← Back to Classes
      </button>

      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-5xl">{classData.icon}</span>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{classData.name}</h1>
              <p className="text-gray-600">4th Grade</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500 mb-1">Your Progress</p>
            <p className="text-3xl font-bold text-gray-900">{classData.progress}%</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            📝 My Notes
          </h3>
          <div className="space-y-3">
            {classNotes.map(note => (
              <div key={note.id} className="p-4 rounded-lg bg-gray-50 hover:bg-gray-100 cursor-pointer">
                <h4 className="font-medium text-gray-900 mb-1">{note.title}</h4>
                <p className="text-sm text-gray-600 mb-2">{note.preview}</p>
                <p className="text-xs text-gray-500">{note.date}</p>
              </div>
            ))}
            <button className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg text-gray-500 hover:border-indigo-300 hover:text-indigo-600 transition-colors flex items-center justify-center gap-2">
              <Plus className="w-4 h-4" />
              Add New Note
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            📅 Upcoming in {classData.name}
          </h3>
          <div className="space-y-3">
            {classEvents.map((event, i) => (
              <div key={i} className="p-4 rounded-lg bg-gray-50">
                <div className="flex items-start justify-between mb-2">
                  <p className="font-medium text-gray-900">{event.event}</p>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    event.type === 'quiz' ? 'bg-yellow-100 text-yellow-800' :
                    event.type === 'assignment' ? 'bg-blue-100 text-blue-800' :
                    event.type === 'project' ? 'bg-purple-100 text-purple-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {event.type}
                  </span>
                </div>
                <p className="text-sm text-gray-500">{event.date}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function NotesView() {
  const [selectedClass, setSelectedClass] = useState('all');

  const filteredNotes = selectedClass === 'all' 
    ? DEMO_NOTES 
    : DEMO_NOTES.filter(note => note.class === selectedClass);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Notes 📝</h1>
        <p className="text-gray-600">All your study notes in one place</p>
      </div>

      <div className="mb-6 flex gap-3">
        <button
          onClick={() => setSelectedClass('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedClass === 'all'
              ? 'bg-indigo-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-50'
          }`}
        >
          All Notes
        </button>
        {DEMO_CLASSES.map(cls => (
          <button
            key={cls.id}
            onClick={() => setSelectedClass(cls.name)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedClass === cls.name
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            {cls.icon} {cls.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {filteredNotes.map(note => (
          <div key={note.id} className="bg-white rounded-xl shadow p-6 hover:shadow-lg transition-all cursor-pointer">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-lg font-bold text-gray-900">{note.title}</h3>
              <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">{note.class}</span>
            </div>
            <p className="text-gray-600 mb-3">{note.preview}</p>
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">{note.date}</p>
              <button className="text-indigo-600 hover:text-indigo-700 text-sm font-medium">
                View Note →
              </button>
            </div>
          </div>
        ))}
        
        <button className="bg-white rounded-xl shadow p-6 border-2 border-dashed border-gray-300 hover:border-indigo-300 transition-colors flex flex-col items-center justify-center gap-3 h-48">
          <Plus className="w-8 h-8 text-gray-400" />
          <span className="text-gray-600 font-medium">Create New Note</span>
        </button>
      </div>
    </div>
  );
}

function CalendarView() {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Calendar 📅</h1>
        <p className="text-gray-600">See what's coming up this week</p>
      </div>

      <div className="bg-white rounded-xl shadow p-6 mb-6">
        <div className="grid grid-cols-7 gap-4 mb-4">
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
            <div key={day} className="text-center font-bold text-gray-600 text-sm">
              {day}
            </div>
          ))}
        </div>
        <div className="grid grid-cols-7 gap-4">
          {[...Array(35)].map((_, i) => {
            const day = i - 1;
            const hasEvent = DEMO_CALENDAR.some(e => parseInt(e.date.split(' ')[1]) === day);
            return (
              <div
                key={i}
                className={`aspect-square flex items-center justify-center rounded-lg ${
                  day === 19 ? 'bg-indigo-600 text-white font-bold' :
                  hasEvent ? 'bg-indigo-50 text-indigo-600 font-medium' :
                  day > 0 ? 'bg-gray-50 text-gray-900' :
                  'bg-transparent'
                }`}
              >
                {day > 0 && day}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Upcoming Events</h3>
        <div className="space-y-3">
          {DEMO_CALENDAR.map((event, i) => (
            <div key={i} className="flex items-start gap-4 p-4 rounded-lg bg-gray-50">
              <div className="text-center min-w-[60px]">
                <div className="text-xs font-medium text-gray-500">{event.date.split(' ')[0]}</div>
                <div className="text-2xl font-bold text-gray-900">{event.date.split(' ')[1]}</div>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-medium text-gray-900">{event.event}</h4>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    event.type === 'quiz' ? 'bg-yellow-100 text-yellow-800' :
                    event.type === 'assignment' ? 'bg-blue-100 text-blue-800' :
                    event.type === 'exam' ? 'bg-red-100 text-red-800' :
                    event.type === 'project' ? 'bg-purple-100 text-purple-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {event.type}
                  </span>
                </div>
                <p className="text-sm text-gray-500">{event.class}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LearnMode({ session }: any) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<any[]>([
    {
      role: 'assistant',
      content: "Hi! I'm your AI learning buddy! 🤖 What would you like to learn about today?"
    }
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    
    setTimeout(() => {
      const responses = [
        "That's a great question! Let me help you understand this better. 🌟",
        "Good thinking! Let me ask you something first: What do you already know about this?",
        "I love your curiosity! Let's explore this together. Here's a fun way to think about it...",
        "You're on the right track! Let me give you a hint to help you figure it out yourself...",
        "Awesome question! This reminds me of something you might have learned before. Do you remember when we talked about..."
      ];
      const aiMsg = {
        role: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)]
      };
      setMessages(prev => [...prev, aiMsg]);
    }, 1000);
    
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!session) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Brain className="w-16 h-16 text-indigo-300 mx-auto mb-4" />
          <p className="text-gray-500 text-lg">Start a learning session from the home page!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden flex flex-col h-[calc(100vh-8rem)]">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="w-6 h-6 text-white" />
              <div>
                <h2 className="text-white font-bold">Learning with AI! 🤖</h2>
                <p className="text-indigo-100 text-sm">● Your AI buddy is here to help</p>
              </div>
            </div>
            <div className="flex gap-2">
              <span className="px-3 py-1 rounded-full bg-indigo-500 text-white text-sm">
                {session.duration} min
              </span>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center flex-shrink-0">
                  <Brain className="w-6 h-6 text-white" />
                </div>
              )}
              <div
                className={`max-w-2xl rounded-2xl p-4 ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="text-lg">{msg.content}</p>
              </div>
              {msg.role === 'user' && (
                <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center text-white font-bold flex-shrink-0 text-lg">
                  A
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your question here..."
              className="flex-1 px-4 py-3 rounded-lg border-2 border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-lg"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-6 h-6" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ReadingMode() {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 bg-white rounded-xl shadow p-8">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Charlotte's Web</h1>
            <p className="text-sm text-gray-500">Chapter 4 • E.B. White</p>
          </div>
          <div className="prose prose-lg max-w-none">
            <p className="text-gray-700 leading-relaxed mb-4 text-lg">
              Wilbur was what farmers call a spring pig, which simply means that he was born in springtime. When he was five weeks old, Mr. Arable said he was now big enough to sell, and would have to be sold.
            </p>
            <p className="text-gray-700 leading-relaxed text-lg">
              "He's got to go, Fern," he said. "You have had your fun raising a baby pig, but Wilbur is not a baby any longer and he has got to be sold."
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              🎯 Reading Tools
            </h3>
            <div className="space-y-3">
              <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-3 border-2 border-gray-200">
                <Lightbulb className="w-5 h-5 text-yellow-500" />
                <span className="font-medium">Explain This</span>
              </button>
              <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-3 border-2 border-gray-200">
                <MessageSquare className="w-5 h-5 text-indigo-500" />
                <span className="font-medium">Ask Question</span>
              </button>
              <button className="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-3 border-2 border-gray-200">
                <FileText className="w-5 h-5 text-purple-500" />
                <span className="font-medium">Summary</span>
              </button>
            </div>
          </div>

          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6 border-2 border-indigo-100">
            <div className="flex items-start gap-3">
              <Brain className="w-6 h-6 text-indigo-600 flex-shrink-0 mt-1" />
              <div>
                <p className="text-sm font-bold text-gray-900 mb-2">💡 Reading Tip</p>
                <p className="text-sm text-gray-700">
                  Notice how the author shows Wilbur is growing up. What does "spring pig" mean? Think about why Mr. Arable wants to sell Wilbur now.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow p-6">
            <h3 className="font-bold text-gray-900 mb-3">📖 Reading Progress</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Chapter 4</span>
                <span className="font-medium text-gray-900">62%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-indigo-500 h-2 rounded-full" style={{ width: '62%' }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function AnalyticsView() {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">My Progress 📊</h1>
        <p className="text-gray-600">See how you're doing in all your classes!</p>
      </div>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Flame className="w-6 h-6 text-orange-500" />
            <p className="text-sm font-medium text-gray-600">Learning Streak</p>
          </div>
          <p className="text-4xl font-bold text-gray-900 mb-1">12 days</p>
          <p className="text-sm text-green-600">🔥 Keep it up!</p>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="w-6 h-6 text-indigo-500" />
            <p className="text-sm font-medium text-gray-600">Time Learning</p>
          </div>
          <p className="text-4xl font-bold text-gray-900 mb-1">8.5 hrs</p>
          <p className="text-sm text-gray-500">This week</p>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Star className="w-6 h-6 text-yellow-500" />
            <p className="text-sm font-medium text-gray-600">Stars Earned</p>
          </div>
          <p className="text-4xl font-bold text-gray-900 mb-1">156</p>
          <p className="text-sm text-green-600">+23 this week</p>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <div className="flex items-center gap-3 mb-2">
            <Award className="w-6 h-6 text-purple-500" />
            <p className="text-sm font-medium text-gray-600">Achievements</p>
          </div>
          <p className="text-4xl font-bold text-gray-900 mb-1">8</p>
          <p className="text-sm text-gray-500">Earned total</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">📚 Class Progress</h3>
          <div className="space-y-4">
            {DEMO_CLASSES.map(cls => (
              <div key={cls.id}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    <span>{cls.icon}</span>
                    {cls.name}
                  </span>
                  <span className="text-sm text-gray-900 font-bold">{cls.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className={`${cls.color} h-2.5 rounded-full transition-all`}
                    style={{ width: `${cls.progress}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">🎯 What I Like to Learn</h3>
          <div className="space-y-4">
            {[
              { activity: 'AI Chat Learning', percent: 45, color: 'bg-indigo-500', emoji: '🤖' },
              { activity: 'Practice Problems', percent: 30, color: 'bg-purple-500', emoji: '✏️' },
              { activity: 'Reading Books', percent: 15, color: 'bg-pink-500', emoji: '📖' },
              { activity: 'Videos & Pictures', percent: 10, color: 'bg-blue-500', emoji: '🎨' },
            ].map(item => (
              <div key={item.activity}>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    <span>{item.emoji}</span>
                    {item.activity}
                  </span>
                  <span className="text-sm text-gray-500">{item.percent}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className={`${item.color} h-2.5 rounded-full transition-all`}
                    style={{ width: `${item.percent}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">🏆 My Achievements</h3>
        <div className="grid grid-cols-3 gap-4">
          {DEMO_ACHIEVEMENTS.map((achievement, i) => (
            <div key={i} className={`p-6 rounded-xl ${achievement.color} text-center`}>
              <div className="text-5xl mb-3">{achievement.icon}</div>
              <h4 className="font-bold text-gray-900 mb-1">{achievement.title}</h4>
              <p className="text-sm text-gray-600">{achievement.description}</p>
            </div>
          ))}
          <div className="p-6 rounded-xl bg-gray-100 text-center border-2 border-dashed border-gray-300">
            <div className="text-5xl mb-3">🔒</div>
            <h4 className="font-bold text-gray-500 mb-1">Next Achievement</h4>
            <p className="text-sm text-gray-500">Keep learning to unlock!</p>
          </div>
        </div>
      </div>
    </div>
  );
}