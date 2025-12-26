type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

interface LoginPageProps {
  onNavigate: (page: Page) => void;
}

export function LoginPage({ onNavigate }: LoginPageProps) {
  return (
    <div className="max-w-md mx-auto px-6 py-16">
      <div className="bg-card border border-border rounded-lg p-8">
        <h2 className="mb-2 text-center">Вход в AILCE</h2>
        <p className="text-muted-foreground text-center mb-8">
          Войдите в свой аккаунт инженера слаботочных систем на базе ИИ
        </p>

        <form className="space-y-6">
          <div>
            <label htmlFor="email" className="block mb-2">
              Адрес электронной почты
            </label>
            <input
              id="email"
              type="email"
              placeholder="your@email.com"
              className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div>
            <label htmlFor="password" className="block mb-2">
              Пароль
            </label>
            <input
              id="password"
              type="password"
              placeholder="Введите ваш пароль"
              className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2">
              <input type="checkbox" className="rounded" />
              <span className="text-sm">Запомнить меня</span>
            </label>
            <button type="button" className="text-sm text-primary hover:underline">
              Забыли пароль?
            </button>
          </div>

          <button
            type="submit"
            className="w-full px-6 py-3 bg-primary text-primary-foreground rounded hover:opacity-90"
          >
            Войти
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-muted-foreground">
            Нет аккаунта?{' '}
            <button
              onClick={() => onNavigate('registration')}
              className="text-primary hover:underline"
            >
              Зарегистрироваться
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}