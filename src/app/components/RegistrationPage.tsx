type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard';

interface RegistrationPageProps {
  onNavigate: (page: Page) => void;
}

export function RegistrationPage({ onNavigate }: RegistrationPageProps) {
  return (
    <div className="max-w-md mx-auto px-6 py-16">
      <div className="bg-card border border-border rounded-lg p-8">
        <h2 className="mb-2 text-center">Создайте свой аккаунт</h2>
        <p className="text-muted-foreground text-center mb-8">
          Присоединяйтесь к AILCE и начните оптимизировать свои проекты
        </p>

        <form className="space-y-6">
          <div>
            <label htmlFor="fullname" className="block mb-2">
              Полное имя
            </label>
            <input
              id="fullname"
              type="text"
              placeholder="Иван Иванов"
              className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

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
            <label htmlFor="company" className="block mb-2">
              Компания (необязательно)
            </label>
            <input
              id="company"
              type="text"
              placeholder="Название вашей компании"
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
              placeholder="Создайте надежный пароль"
              className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div>
            <label htmlFor="confirm-password" className="block mb-2">
              Подтверждение пароля
            </label>
            <input
              id="confirm-password"
              type="password"
              placeholder="Введите пароль еще раз"
              className="w-full px-4 py-2 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div>
            <label className="flex items-start gap-2">
              <input type="checkbox" className="mt-1 rounded" />
              <span className="text-sm text-muted-foreground">
                Я согласен с Условиями использования и Политикой конфиденциальности
              </span>
            </label>
          </div>

          <button
            type="submit"
            className="w-full px-6 py-3 bg-primary text-primary-foreground rounded hover:opacity-90"
          >
            Создать аккаунт
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-muted-foreground">
            Уже есть аккаунт?{' '}
            <button
              onClick={() => onNavigate('login')}
              className="text-primary hover:underline"
            >
              Войти
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}