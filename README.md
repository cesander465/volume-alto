# Volume Alto

Recomendador musical feito com Flask, pandas, NumPy e scikit-learn.

## Executar localmente

Requer Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Acesse http://127.0.0.1:5000.

## Publicar na Vercel

1. Envie o conteúdo desta pasta para um repositório GitHub.
2. Na Vercel, escolha **Add New Project** e importe o repositório.
3. Mantenha as configurações de build padrão (não é necessário Build Command nem Output Directory).
4. Clique em **Deploy**.

A Vercel detecta o aplicativo Flask pelo objeto `app` em `app.py`. Os arquivos estáticos ficam em `public/` e o CSS é servido em `/style.css`.
