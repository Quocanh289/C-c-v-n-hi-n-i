export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold">Mental Health Text Analyzer</h1>
      </div>
      <div className="relative flex place-items-center">
        <textarea
          className="w-full h-32 p-4 border rounded"
          placeholder="Enter your text here..."
        ></textarea>
      </div>
      <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Analyze
      </button>
    </main>
  )
}