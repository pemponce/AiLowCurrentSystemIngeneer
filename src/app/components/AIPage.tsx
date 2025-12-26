import { useState } from 'react';

type Page = 'home' | 'login' | 'registration' | 'subscribe' | 'dashboard' | 'ai' | 'profile';

interface AIPageProps {
  onNavigate: (page: Page) => void;
}

export function AIPage({ onNavigate }: AIPageProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [userPreferences, setUserPreferences] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setUploadedFile(e.dataTransfer.files[0]);
    }
  };

  const handleGenerate = () => {
    if (!uploadedFile) {
      alert('Пожалуйста, загрузите файл');
      return;
    }
    setIsGenerating(true);
    // Имитация процесса генерации
    setTimeout(() => {
      setIsGenerating(false);
      alert('Генерация завершена! В реальном приложении здесь будет результат работы ИИ.');
    }, 3000);
  };

  const removeFile = () => {
    setUploadedFile(null);
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <div className="mb-8">
        <h1 className="mb-2">Нейросеть AILCE</h1>
        <p className="text-muted-foreground">
          Загрузите ваши чертежи и опишите требования для автоматической генерации проектной документации
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Левая колонка - Загрузка файла */}
        <div>
          <h2 className="mb-4">Загрузка файлов</h2>
          
          {/* Drag & Drop область */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? 'border-primary bg-primary/5'
                : 'border-border bg-card hover:border-primary/50'
            }`}
          >
            {!uploadedFile ? (
              <div>
                <div className="text-6xl mb-4">📁</div>
                <h3 className="mb-2">Перетащите файл сюда</h3>
                <p className="text-muted-foreground mb-4">
                  или нажмите кнопку ниже для выбора
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Поддерживаемые форматы: PNG, PDF, DWG
                </p>
                <label className="inline-block px-6 py-3 bg-primary text-primary-foreground rounded hover:opacity-90 cursor-pointer">
                  Выбрать файл
                  <input
                    type="file"
                    accept=".png,.pdf,.dwg"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <div>
                <div className="text-6xl mb-4">✅</div>
                <h3 className="mb-2">Файл загружен</h3>
                <p className="text-muted-foreground mb-4">{uploadedFile.name}</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Размер: {(uploadedFile.size / 1024 / 1024).toFixed(2)} МБ
                </p>
                <button
                  onClick={removeFile}
                  className="px-4 py-2 border border-border rounded hover:bg-accent"
                >
                  Удалить файл
                </button>
              </div>
            )}
          </div>

          {/* Информация о файле */}
          {uploadedFile && (
            <div className="mt-6 bg-muted p-4 rounded-lg">
              <h4 className="mb-2">Информация о файле</h4>
              <div className="space-y-1 text-sm">
                <p><span className="text-muted-foreground">Имя:</span> {uploadedFile.name}</p>
                <p><span className="text-muted-foreground">Тип:</span> {uploadedFile.type || 'Неизвестно'}</p>
                <p><span className="text-muted-foreground">Размер:</span> {(uploadedFile.size / 1024 / 1024).toFixed(2)} МБ</p>
              </div>
            </div>
          )}
        </div>

        {/* Правая колонка - Настройки генерации */}
        <div>
          <h2 className="mb-4">Настройки генерации</h2>
          
          <div className="bg-card border border-border rounded-lg p-6">
            <label htmlFor="preferences" className="block mb-2">
              Опишите ваши требования
            </label>
            <textarea
              id="preferences"
              value={userPreferences}
              onChange={(e) => setUserPreferences(e.target.value)}
              placeholder="Например: Необходимо создать спецификацию оборудования для системы видеонаблюдения, включая камеры, регистраторы и кабельную продукцию. Объект: офисное здание на 5 этажей..."
              rows={12}
              className="w-full px-4 py-3 bg-input-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring resize-none"
            />
            <p className="text-sm text-muted-foreground mt-2">
              Опишите детально, что именно вы хотите получить на выходе
            </p>
          </div>

          {/* Дополнительные опции */}
          <div className="mt-6 bg-card border border-border rounded-lg p-6">
            <h3 className="mb-4">Дополнительные параметры</h3>
            <div className="space-y-4">
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" defaultChecked />
                <span>Включить спецификацию оборудования</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" defaultChecked />
                <span>Рассчитать стоимость материалов</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span>Создать план прокладки кабелей</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span>Добавить монтажные схемы</span>
              </label>
            </div>
          </div>

          {/* Кнопка генерации */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !uploadedFile}
            className={`w-full mt-6 px-6 py-4 rounded transition-opacity ${
              isGenerating || !uploadedFile
                ? 'bg-muted text-muted-foreground cursor-not-allowed'
                : 'bg-primary text-primary-foreground hover:opacity-90'
            }`}
          >
            {isGenerating ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin">⚙️</span>
                Генерация... Пожалуйста, подождите
              </span>
            ) : (
              '🚀 Начать генерацию'
            )}
          </button>

          {!uploadedFile && (
            <p className="text-sm text-muted-foreground text-center mt-3">
              Сначала загрузите файл для начала работы
            </p>
          )}
        </div>
      </div>

      {/* Пример результатов */}
      <div className="mt-12 bg-muted p-6 rounded-lg">
        <h2 className="mb-4">Что вы получите:</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-card p-4 rounded border border-border">
            <div className="text-2xl mb-2">📄</div>
            <h4 className="mb-1">Проектная документация</h4>
            <p className="text-sm text-muted-foreground">
              Полный комплект документов в соответствии с ГОСТ
            </p>
          </div>
          <div className="bg-card p-4 rounded border border-border">
            <div className="text-2xl mb-2">📊</div>
            <h4 className="mb-1">Спецификации</h4>
            <p className="text-sm text-muted-foreground">
              Детальные спецификации всего оборудования
            </p>
          </div>
          <div className="bg-card p-4 rounded border border-border">
            <div className="text-2xl mb-2">💰</div>
            <h4 className="mb-1">Смета</h4>
            <p className="text-sm text-muted-foreground">
              Расчет стоимости материалов и работ
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
