from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# Примитив: просто точки и линии в миллиметрах, без масштабов (MVP)
def export_pdf(project_id: str, rooms, devices, routes, out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    w, h = A4
    c.setLineWidth(0.5)
    # Рисуем комнаты
    c.setStrokeColorRGB(0, 0, 0)
    for room in rooms:
        pts = [(x * cm, y * cm) for x, y in list(room.exterior.coords)]
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
    # Устройства
    for t, _, p in devices:
        x, y = p.x * cm, p.y * cm
        c.circle(x, y, 0.15 * cm, stroke=1, fill=0)
    # Маршруты
    c.setStrokeColorRGB(0.2, 0.2, 0.8)
    for _, ls, _ in routes:
        coords = [(x * cm, y * cm) for x, y in list(ls.coords)]
        for i in range(len(coords) - 1):
            c.line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    c.showPage()
    c.save()
