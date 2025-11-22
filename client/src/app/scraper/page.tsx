'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Smartphone, Download, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

type Review = {
  content: string;
};

type ScraperState = {
  isLoading: boolean;
  error: string | null;
  reviews: Review[] | null;
  count: number | null;
};

export default function ScraperPage() {
  const [appId, setAppId] = useState('com.google.android.apps.maps');
  const [lang, setLang] = useState('en');
  const [country, setCountry] = useState('us');
  const [state, setState] = useState<ScraperState>({
    isLoading: false,
    error: null,
    reviews: null,
    count: null,
  });

  const handleScrape = async () => {
    if (!appId.trim()) return;

    setState({ isLoading: true, error: null, reviews: null, count: null });

    try {
      const formData = new FormData();
      formData.append('app_id', appId);
      formData.append('lang', lang);
      formData.append('country', country);

      const response = await fetch('http://localhost:8000/scrape-reviews', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();
      setState({
        isLoading: false,
        error: null,
        reviews: data.reviews,
        count: data.count,
      });
    } catch (error) {
      setState({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        reviews: null,
        count: null,
      });
    }
  };

  const downloadCSV = () => {
    if (!state.reviews) return;

    const csvContent = 'content\n' + state.reviews.map(r => `"${r.content.replace(/"/g, '""')}"`).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'reviews.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-2 hover:text-blue-600 transition-colors">
              <Smartphone className="h-8 w-8 text-blue-600" />
              <span className="text-2xl font-bold">ReviewNet</span>
            </Link>
            <nav className="flex space-x-6">
              <Link href="/" className="text-gray-600 hover:text-gray-900 transition-colors">Home</Link>
              <Link href="/sentiment" className="text-gray-600 hover:text-gray-900 transition-colors">Sentiment</Link>
              <Link href="/classification" className="text-gray-600 hover:text-gray-900 transition-colors">Classification</Link>
              <Link href="/analytics" className="text-gray-600 hover:text-gray-900 transition-colors">Analytics</Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          {/* Title */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">Google Play Review Scraper</h1>
            <p className="text-gray-600">
              Extract all reviews from any app on Google Play Store. Simply enter the app ID below.
            </p>
          </div>

          {/* Form */}
          <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
            <div className="space-y-4">
              <div>
                <label htmlFor="appId" className="block text-sm font-medium text-gray-700 mb-2">
                  App ID *
                </label>
                <input
                  id="appId"
                  type="text"
                  value={appId}
                  onChange={(e) => setAppId(e.target.value)}
                  placeholder="com.google.android.apps.maps"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-sm text-gray-500 mt-1">
                  e.g., com.google.android.apps.maps or com.instagram.android
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="lang" className="block text-sm font-medium text-gray-700 mb-2">
                    Language
                  </label>
                  <select
                    id="lang"
                    value={lang}
                    onChange={(e) => setLang(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="it">Italian</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="country" className="block text-sm font-medium text-gray-700 mb-2">
                    Country
                  </label>
                  <select
                    id="country"
                    value={country}
                    onChange={(e) => setCountry(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="us">United States</option>
                    <option value="gb">United Kingdom</option>
                    <option value="ca">Canada</option>
                    <option value="au">Australia</option>
                    <option value="de">Germany</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handleScrape}
                disabled={state.isLoading || !appId.trim()}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                {state.isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Scraping reviews...
                  </>
                ) : (
                  <>
                    <Smartphone className="w-5 h-5 mr-2" />
                    Scrape Reviews
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results */}
          {state.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
                <div>
                  <h3 className="text-red-800 font-medium">Error</h3>
                  <p className="text-red-700 text-sm">{state.error}</p>
                </div>
              </div>
            </div>
          )}

          {state.reviews && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <CheckCircle className="w-5 h-5 text-green-600 mr-3" />
                  <div>
                    <h3 className="text-green-800 font-medium">Success!</h3>
                    <p className="text-green-700 text-sm">
                      Scraped {state.count} reviews from {appId}
                    </p>
                  </div>
                </div>
                <button
                  onClick={downloadCSV}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </button>
              </div>

              {/* Preview */}
              <div className="mt-4">
                <h4 className="font-medium text-green-800 mb-2">Preview</h4>
                <div className="bg-white rounded border p-3 max-h-40 overflow-y-auto text-sm">
                  {state.reviews.slice(0, 3).map((review, index) => (
                    <div key={index} className="mb-2 last:mb-0 pb-2 last:pb-0 border-b last:border-b-0">
                      "{review.content.length > 100 ? review.content.slice(0, 100) + '...' : review.content}"
                    </div>
                  ))}
                  {state.reviews.length > 3 && <div className="text-gray-500">... and {state.reviews.length - 3} more</div>}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
