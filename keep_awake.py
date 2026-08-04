#!/usr/bin/env python3
"""
keep_awake.py — Mantém um app do Streamlit Community Cloud acordado.

Por que não basta um "ping" HTTP: o Streamlit responde 200 com um shell HTML
estático e só inicia a sessão real quando o JavaScript roda e abre o WebSocket
(/_stcore/stream). Portanto usamos um navegador headless (Chromium via
Playwright): abrimos o app, clicamos no botão de acordar se ele estiver
dormindo, e permanecemos alguns segundos para a sessão ser registrada como
tráfego real.
"""

import os
import sys

from playwright.sync_api import sync_playwright

URL = os.environ.get("STREAMLIT_URL", "").strip()

# Texto do botão da página "This app has gone to sleep" (mantido amplo por
# robustez, caso o Streamlit mude a frase exata).
WAKE_TEXTS = [
    "Yes, get this app back up!",
    "get this app back up",
    "back up",
]

# Tempo (s) que permanecemos na página para a sessão WebSocket registrar.
DWELL_SECONDS = int(os.environ.get("KEEP_AWAKE_DWELL", "25"))


def main() -> int:
    if not URL:
        print("ERRO: defina a variável STREAMLIT_URL "
              "(ex.: https://seu-app.streamlit.app).")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Visitando {URL} ...")
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception as e:
            print(f"ERRO ao carregar o app: {e}")
            browser.close()
            return 1

        # Dá um momento para a página de "dormindo" renderizar (se for o caso).
        page.wait_for_timeout(5_000)

        clicked = False
        for txt in WAKE_TEXTS:
            try:
                btn = page.get_by_text(txt, exact=False)
                if btn.count() > 0 and btn.first.is_visible():
                    btn.first.click(timeout=10_000)
                    clicked = True
                    print("App estava dormindo — botão de acordar clicado.")
                    break
            except Exception:
                continue

        if not clicked:
            print("Nenhum botão de acordar encontrado — app já estava acordado.")

        # Permanece na página para a sessão contar como tráfego real.
        page.wait_for_timeout(DWELL_SECONDS * 1000)

        try:
            print(f"Título da página: {page.title()!r}")
        except Exception:
            pass

        browser.close()

    print("Concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
