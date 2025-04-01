export default function ExerciseCard({ question, options, onAnswer }) {
  return (
    <div className="bg-white shadow-xl rounded-xl p-6">
      <h2 className="text-2xl font-semibold mb-4">{question}</h2>
      {options.map((option, idx) => (
        <button
          key={idx}
          className="block w-full text-left px-4 py-2 border rounded-lg mb-2 hover:bg-blue-50"
          onClick={() => onAnswer(option)}
        >
          {option}
        </button>
      ))}
    </div>
  );
}
