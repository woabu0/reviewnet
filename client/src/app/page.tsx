'use client';

import Link from "next/link";
import { Smartphone, MessageCircle, BarChart3, Database, ArrowRight } from "lucide-react";

const features = [
  {
    icon: Smartphone,
    title: "Review Scraper",
    description: "Extract reviews from Google Play Store with ease. Enter an app ID and get all reviews in seconds.",
    href: "/scraper",
    color: "bg-blue-50 text-blue-600 border-blue-200"
  },
  {
    icon: MessageCircle,
    title: "Sentiment Analysis",
    description: "Analyze customer sentiment with multilingual support. Upload CSV files and get detailed insights.",
    href: "/sentiment",
    color: "bg-green-50 text-green-600 border-green-200"
  },
  {
    icon: BarChart3,
    title: "Thematic Classification",
    description: "Categorize reviews into themes using advanced ML models. Perfect for accessibility app reviews.",
    href: "/classification",
    color: "bg-yellow-50 text-yellow-600 border-yellow-200"
  },
  {
    icon: Database,
    title: "Analytics Dashboard",
    description: "Visualize your data with beautiful charts and insights. Make data-driven decisions.",
    href: "/analytics",
    color: "bg-purple-50 text-purple-600 border-purple-200"
  }
];

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">ReviewNet</h1>
            </div>
            <nav className="hidden md:flex space-x-6">
              <Link href="/" className="text-gray-600 hover:text-gray-900 transition-colors">
                Home
              </Link>
              <Link href="/scraper" className="text-gray-600 hover:text-gray-900 transition-colors">
                Scraper
              </Link>
              <Link href="/sentiment" className="text-gray-600 hover:text-gray-900 transition-colors">
                Sentiment
              </Link>
              <Link href="/classification" className="text-gray-600 hover:text-gray-900 transition-colors">
                Classification
              </Link>
              <Link href="/analytics" className="text-gray-600 hover:text-gray-900 transition-colors">
                Analytics
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto mb-16">
          <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            AI-Powered App Review
            <span className="text-blue-600 block">Analytics Platform</span>
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Extract, analyze, and understand customer feedback from Google Play Store reviews.
            Multilingual sentiment analysis and thematic classification powered by machine learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/scraper"
              className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors inline-flex items-center"
            >
              Get Started <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
            <Link
              href="/analytics"
              className="bg-white text-gray-700 border border-gray-300 px-8 py-3 rounded-lg font-medium hover:bg-gray-50 transition-colors inline-flex items-center"
            >
              View Examples
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature) => (
            <Link key={feature.title} href={feature.href}>
              <div className={`p-6 rounded-xl border transition-all hover:shadow-lg hover:-translate-y-1 ${feature.color}`}>
                <feature.icon className="h-12 w-12 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
                <div className="mt-4 flex items-center text-sm font-medium">
                  Try now <ArrowRight className="ml-1 h-3 w-3" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-white py-16 border-t">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-blue-600 mb-2">100+</div>
              <div className="text-gray-600">Apps Analyzed</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-green-600 mb-2">10k+</div>
              <div className="text-gray-600">Reviews Processed</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-purple-600 mb-2">5+</div>
              <div className="text-gray-600">Theme Categories</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; 2024 ReviewNet. Analyze app reviews with AI power.</p>
        </div>
      </footer>
    </div>
  );
}
