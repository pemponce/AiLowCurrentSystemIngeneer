type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

interface HomePageProps {
  onNavigate: (page: Page) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <div className="max-w-7xl mx-auto px-6 py-16">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="mb-6">Добро пожаловать в AILCE</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto mb-8">
          Инженер слаботочных систем на базе ИИ - Ваше интеллектуальное решение для проектирования слаботочных электрических систем. 
          Мы предоставляем передовые инструменты и услуги на основе искусственного интеллекта для современного проектирования и анализа электрических систем.
        </p>
        <div className="flex gap-4 justify-center">
          <button 
            onClick={() => onNavigate('registration')}
            className="px-6 py-3 bg-primary text-primary-foreground rounded hover:opacity-90"
          >
            Начать работу
          </button>
          <button 
            onClick={() => onNavigate('subscribe')}
            className="px-6 py-3 border border-border rounded hover:bg-accent"
          >
            Посмотреть тарифы
          </button>
        </div>
      </div>

      {/* About Section */}
      <div className="grid md:grid-cols-3 gap-8 mb-16">
        <div className="p-6 border border-border rounded-lg">
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🤖</span>
          </div>
          <h3 className="mb-2">На базе ИИ</h3>
          <p className="text-muted-foreground">
            Передовые алгоритмы искусственного интеллекта для оптимизации ваших проектов в области электротехники.
          </p>
        </div>

        <div className="p-6 border border-border rounded-lg">
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">⚡</span>
          </div>
          <h3 className="mb-2">Слаботочные системы</h3>
          <p className="text-muted-foreground">
            Специализация на проектировании систем передачи данных, безопасности, пожарной сигнализации и связи.
          </p>
        </div>

        <div className="p-6 border border-border rounded-lg">
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">📊</span>
          </div>
          <h3 className="mb-2">Умный анализ</h3>
          <p className="text-muted-foreground">
            Анализ систем в реальном времени и рекомендации для оптимальной производительности.
          </p>
        </div>
      </div>

      {/* Company Info */}
      <div className="bg-muted p-8 rounded-lg">
        <h2 className="mb-4">О нашей компании</h2>
        <p className="text-muted-foreground mb-4">
          AILCE объединяет многолетний опыт в области электротехники с новейшими технологиями искусственного интеллекта 
          для предоставления инновационных решений для слаботочных систем. Наша платформа помогает инженерам 
          проектировать, анализировать и оптимизировать электрические системы с беспрецедентной скоростью и точностью.
        </p>
        <p className="text-muted-foreground">
          Независимо от того, работаете ли вы над коммерческими зданиями, жилыми проектами или промышленными объектами, 
          AILCE предоставляет инструменты, необходимые для успеха в современной электротехнике.
        </p>
      </div>
    </div>
  );
}