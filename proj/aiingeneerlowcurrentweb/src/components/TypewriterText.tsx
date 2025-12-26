import { useState, useEffect } from 'react';

interface TypewriterTextProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
}

export function TypewriterText({ 
  text, 
  speed = 100, 
  delay = 0,
  className = '' 
}: TypewriterTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    // Reset when text changes
    setDisplayedText('');
    setCurrentIndex(0);
    setIsTyping(false);

    // Start typing after delay
    const delayTimeout = setTimeout(() => {
      setIsTyping(true);
    }, delay);

    return () => clearTimeout(delayTimeout);
  }, [text, delay]);

  useEffect(() => {
    if (!isTyping || currentIndex >= text.length) {
      return;
    }

    const timeout = setTimeout(() => {
      setDisplayedText(text.slice(0, currentIndex + 1));
      setCurrentIndex(currentIndex + 1);
    }, speed);

    return () => clearTimeout(timeout);
  }, [currentIndex, isTyping, text, speed]);

  const isComplete = currentIndex >= text.length && isTyping;

  return (
    <span className={className}>
      {displayedText}
      <span 
        className={`inline-block w-0.5 h-[1em] bg-current ml-0.5 align-middle ${
          isComplete ? 'animate-blink' : 'opacity-100'
        }`}
        style={{ animation: isComplete ? 'blink 1s step-end infinite' : 'none' }}
      />
    </span>
  );
}
