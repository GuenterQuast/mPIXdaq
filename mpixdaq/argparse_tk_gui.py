"""
argparse_tk_gui
================

Eine kleine, abhängigkeitsfreie Python-Bibliothek, die aus einem bestehenden
``argparse.ArgumentParser`` automatisch eine grafische Oberfläche (Tkinter/TTK)
erzeugt.

Grundidee
---------
Statt ein CLI-Programm nur über die Kommandozeile zu bedienen, kann man mit
dieser Bibliothek aus demselben ``ArgumentParser``-Objekt ein Formular bauen
lassen. Der Benutzer füllt Felder aus, klickt auf "Ausführen", und die
Bibliothek baut daraus die passende ``sys.argv``-Liste und ruft entweder

  * eine von dir übergebene Funktion ``main(args)`` auf, oder
  * das Skript selbst per Subprozess (z. B. wenn du dein Programm nicht
    importieren willst), oder
  * gibt dir einfach die fertige Argumentliste zurück.

Unterstützte argparse-Feature
------------------------------
- positional und optionale Argumente
- ``type=`` (str, int, float, bool über store_true/store_false)
- ``choices=``  -> Dropdown (Combobox)
- ``action="store_true"/"store_false"`` -> Checkbox
- ``action="append"`` / ``nargs="*"`` / ``nargs="+"`` -> Mehrfachwerte
  (kommagetrennt eingegeben)
- ``required=True`` -> Feld wird visuell markiert und vor dem Start geprüft
- ``default=`` -> wird vorbelegt
- ``help=`` -> wird als Tooltip / Beschriftung angezeigt
- Mutually-exclusive-Groups (zumindest visuell gruppiert)
- Subparser (``add_subparsers``) -> Tabs/Reiter, ein Tab pro Subcommand

Schnellstart
------------
```python
import argparse
from argparse_tk_gui import ArgparseGUI

parser = argparse.ArgumentParser(description="Mein Tool")
parser.add_argument("input", help="Eingabedatei")
parser.add_argument("-o", "--output", default="out.txt", help="Zieldatei")
parser.add_argument("-v", "--verbose", action="store_true", help="Ausführlich")
parser.add_argument("--mode", choices=["fast", "slow"], default="fast")

def main(args):
    print("Würde jetzt laufen mit:", args)

if __name__ == "__main__":
    gui = ArgparseGUI(parser, run_callback=main)
    gui.mainloop()
```

Lizenz: frei verwendbar (MIT-artig), keine externen Abhängigkeiten außer der
Python-Standardbibliothek (tkinter, argparse, subprocess, sys, shlex).

     Code erzeugt mit Claude AI, Juni 2026

"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import Any, Callable, Optional


__all__ = ["ArgparseGUI", "build_gui", "FieldSpec"]
__version__ = "1.0.0"


# --------------------------------------------------------------------------
# Hilfsklassen
# --------------------------------------------------------------------------


class FieldSpec:
    """Interne Repräsentation eines einzelnen argparse-Arguments für die GUI."""

    def __init__(self, action: argparse.Action):
        self.action = action
        self.dest = action.dest
        self.option_strings = action.option_strings
        self.is_positional = len(action.option_strings) == 0
        self.help = action.help or ""
        self.required = getattr(action, "required", False) or self.is_positional
        self.default = action.default
        self.choices = list(action.choices) if action.choices else None
        self.nargs = action.nargs
        self.action_type = type(action).__name__  # z.B. _StoreTrueAction
        self.type_func = action.type
        self.var: Optional[tk.Variable] = None
        self.widget: Optional[tk.Widget] = None

    @property
    def label(self) -> str:
        if self.is_positional:
            return self.dest
        # Bevorzugt die "lange" Option als Label
        long_opts = [o for o in self.option_strings if o.startswith("--")]
        return long_opts[0] if long_opts else self.option_strings[0]

    @property
    def is_bool_flag(self) -> bool:
        return self.action_type in ("_StoreTrueAction", "_StoreFalseAction")

    @property
    def is_multi(self) -> bool:
        return (
            self.action_type == "_AppendAction"
            or self.nargs in ("*", "+")
            or (isinstance(self.nargs, int) and self.nargs > 1)
        )

    @property
    def is_count(self) -> bool:
        return self.action_type == "_CountAction"

    @property
    def is_help_action(self) -> bool:
        return self.action_type == "_HelpAction"


# --------------------------------------------------------------------------
# Tooltip (kleine Eigenimplementierung, keine externe Lib nötig)
# --------------------------------------------------------------------------


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip: Optional[tk.Toplevel] = None
        if not text:
            return
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 9),
            wraplength=320,
        )
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


# --------------------------------------------------------------------------
# Hauptklasse
# --------------------------------------------------------------------------


class ArgparseGUI:
    """
    Baut aus einem ``argparse.ArgumentParser`` ein Tkinter-Fenster.

    Parameter
    ---------
    parser:
        Der bereits konfigurierte ArgumentParser (mit add_argument(...) Aufrufen).
    run_callback:
        Optionale Funktion ``f(args: argparse.Namespace)``, die beim Klick auf
        "Ausführen" mit dem geparsten Namespace aufgerufen wird. Wird sie nicht
        angegeben, ruft die GUI stattdessen ``script_path`` als Subprozess auf
        (falls gesetzt), oder zeigt nur die erzeugte Befehlszeile an.
    script_path:
        Pfad zu einem Python-Skript, das per ``subprocess`` mit den erzeugten
        Argumenten aufgerufen wird (Alternative zu run_callback).
    title:
        Fenstertitel. Default: parser.prog.
    capture_output:
        Wenn True (Default) und ein Subprozess verwendet wird, wird dessen
        stdout/stderr in einem Textfeld im Fenster angezeigt.
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        run_callback: Optional[Callable[[argparse.Namespace], Any]] = None,
        script_path: Optional[str] = None,
        title: Optional[str] = None,
        capture_output: bool = True,
        theme: Optional[str] = None,
    ):
        self.parser = parser
        self.run_callback = run_callback
        self.script_path = script_path
        self.capture_output = capture_output

        self.root = tk.Tk()
        self.root.title(title or parser.prog or "Argparse GUI")
        # self.root.geometry("640x560")
        # get screen width and height
        ws = self.root.winfo_screenwidth()  # width of the screen
        hs = self.root.winfo_screenheight()  # height of the screen
        w, h = (
            550,
            950,
        )
        x, y = 50, 50
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.minsize(480, 360)

        style = ttk.Style(self.root)
        try:
            style.theme_use(theme or style.theme_use())
        except tk.TclError:
            pass

        self.fields: list[FieldSpec] = []
        self.subparsers_action: Optional[argparse._SubParsersAction] = None
        self.sub_field_groups: dict[str, list[FieldSpec]] = {}

        self._build_layout()

    # ---------------------------------------------------------- öffentliche API

    def mainloop(self):
        self.root.mainloop()

    def build_argv(self) -> list[str]:
        """Liest alle Widgets aus und baut eine sys.argv-kompatible Liste."""
        if self.subparsers_action is not None:
            tab_idx = self.notebook.index(self.notebook.select())
            sub_name = self._tab_names[tab_idx]
            fields = self.sub_field_groups[sub_name]
            argv = [sub_name] + self._collect_argv(fields)
            return argv
        return self._collect_argv(self.fields)

    def parse_args(self) -> argparse.Namespace:
        """Wertet die Formularfelder aus und gibt einen argparse.Namespace zurück."""
        argv = self.build_argv()
        return self.parser.parse_args(argv)

    # ---------------------------------------------------------- Layout

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        if self.parser.description:
            desc = ttk.Label(outer, text=self.parser.description, wraplength=600, font=("TkDefaultFont", 10, "italic"))
            desc.pack(fill="x", pady=(0, 8))

        # Prüfen ob Subparser vorhanden sind
        sub_action = self._find_subparsers_action(self.parser)

        form_container = ttk.Frame(outer)
        form_container.pack(fill="both", expand=True)

        if sub_action:
            self.subparsers_action = sub_action
            self.notebook = ttk.Notebook(form_container)
            self.notebook.pack(fill="both", expand=True)
            self._tab_names: list[str] = []
            for name, sub_parser in sub_action.choices.items():
                tab = ttk.Frame(self.notebook, padding=10)
                self.notebook.add(tab, text=name)
                self._tab_names.append(name)
                canvas, scroll_frame = self._make_scrollable(tab)
                fields = self._build_fields_for(sub_parser, scroll_frame)
                self.sub_field_groups[name] = fields
        else:
            canvas, scroll_frame = self._make_scrollable(form_container)
            self.fields = self._build_fields_for(self.parser, scroll_frame)

        # Buttonleiste
        btn_bar = ttk.Frame(outer)
        btn_bar.pack(fill="x", pady=(10, 0))

        ttk.Button(btn_bar, text="Befehl anzeigen", command=self._on_show_cmd).pack(side="left")
        ttk.Button(btn_bar, text="Ausführen", command=self._on_run).pack(side="right")
        ttk.Button(btn_bar, text="Beenden", command=self.root.destroy).pack(side="right", padx=(0, 6))

        # Ausgabefenster
        if self.capture_output:
            out_frame = ttk.LabelFrame(outer, text="Ausgabe", padding=4)
            out_frame.pack(fill="both", expand=False, pady=(10, 0))
            self.output_text = scrolledtext.ScrolledText(out_frame, height=8, state="disabled", font=("TkFixedFont", 9))
            self.output_text.pack(fill="both", expand=True)

    def _make_scrollable(self, parent: tk.Widget):
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        return canvas, inner

    @staticmethod
    def _find_subparsers_action(parser: argparse.ArgumentParser):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action
        return None

    # ---------------------------------------------------------- Felder bauen

    def _build_fields_for(self, parser: argparse.ArgumentParser, container: tk.Widget) -> list[FieldSpec]:
        fields: list[FieldSpec] = []
        row = 0
        container.columnconfigure(1, weight=1)

        for action in parser._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            spec = FieldSpec(action)
            fields.append(spec)

            label_text = spec.label + (" *" if spec.required else "")
            label = ttk.Label(container, text=label_text)
            label.grid(row=row, column=0, sticky="ne", padx=(0, 8), pady=4)
            if spec.required:
                label.configure(foreground="#a33")

            widget = self._build_widget(container, spec)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

            if spec.help:
                _Tooltip(label, spec.help)
                _Tooltip(widget, spec.help)
                help_lbl = ttk.Label(
                    container, text=spec.help, foreground="#666", font=("TkDefaultFont", 8), wraplength=380
                )
                row += 1
                help_lbl.grid(row=row, column=1, sticky="w", pady=(0, 4))

            row += 1

        return fields

    def _build_widget(self, parent: tk.Widget, spec: FieldSpec) -> tk.Widget:
        # Checkbox für store_true / store_false
        if spec.is_bool_flag:
            var = tk.BooleanVar(value=bool(spec.default))
            spec.var = var
            cb = ttk.Checkbutton(parent, variable=var)
            spec.widget = cb
            return cb

        # Count-Action -> Spinbox (0..10)
        if spec.is_count:
            var = tk.IntVar(value=spec.default or 0)
            spec.var = var
            sb = ttk.Spinbox(parent, from_=0, to=20, textvariable=var, width=6)
            spec.widget = sb
            return sb

        # Choices -> Combobox
        if spec.choices:
            var = tk.StringVar(value=str(spec.default) if spec.default is not None else "")
            spec.var = var
            cb = ttk.Combobox(parent, textvariable=var, values=[str(c) for c in spec.choices], state="readonly")
            spec.widget = cb
            return cb

        # Datei-/Pfad-Heuristik: dest enthält "file", "path", "dir" -> Entry + Browse-Button
        wrapper = ttk.Frame(parent)
        wrapper.columnconfigure(0, weight=1)

        var = tk.StringVar(value=self._default_to_str(spec.default) if spec.default is not None else "")
        spec.var = var
        entry = ttk.Entry(wrapper, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew")
        spec.widget = entry

        dest_lower = spec.dest.lower()
        if any(k in dest_lower for k in ("file", "path")) and "dir" not in dest_lower:
            btn = ttk.Button(wrapper, text="…", width=3, command=lambda v=var: self._pick_file(v))
            btn.grid(row=0, column=1, padx=(4, 0))
        elif "dir" in dest_lower or "folder" in dest_lower:
            btn = ttk.Button(wrapper, text="…", width=3, command=lambda v=var: self._pick_dir(v))
            btn.grid(row=0, column=1, padx=(4, 0))

        if spec.is_multi:
            hint = ttk.Label(
                wrapper, text="(mehrere Werte: kommagetrennt)", font=("TkDefaultFont", 7), foreground="#888"
            )
            hint.grid(row=1, column=0, columnspan=2, sticky="w")

        return wrapper

    @staticmethod
    def _default_to_str(default) -> str:
        if isinstance(default, (list, tuple)):
            return ", ".join(str(d) for d in default)
        return str(default)

    def _pick_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _pick_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ---------------------------------------------------------- Argumente sammeln

    def _collect_argv(self, fields: list[FieldSpec]) -> list[str]:
        argv: list[str] = []
        missing_required = []

        for spec in fields:
            if spec.var is None:
                continue

            if spec.is_bool_flag:
                checked = spec.var.get()
                default_checked = bool(spec.default)
                if checked != default_checked:
                    argv.append(spec.option_strings[0])
                continue

            raw = spec.var.get()
            if isinstance(raw, str):
                raw = raw.strip()

            is_empty = raw == "" or raw is None
            if is_empty:
                if spec.required:
                    missing_required.append(spec.label)
                continue

            values: list[str]
            if spec.is_multi and isinstance(raw, str):
                values = [v.strip() for v in raw.split(",") if v.strip() != ""]
            else:
                values = [str(raw)]

            if spec.is_positional:
                argv.extend(values)
            else:
                opt = spec.option_strings[0]
                if spec.is_multi:
                    argv.append(opt)
                    argv.extend(values)
                else:
                    argv.append(opt)
                    argv.append(values[0])

        if missing_required:
            raise ValueError("Pflichtfelder fehlen: " + ", ".join(missing_required))

        return argv

    # ---------------------------------------------------------- Button-Handler

    def _on_show_cmd(self):
        try:
            argv = self.build_argv()
        except ValueError as e:
            messagebox.showerror("Fehlende Angaben", str(e))
            return
        cmd = "<prog> " + " ".join(shlex.quote(a) for a in argv)
        messagebox.showinfo("Erzeugter Befehl", cmd)

    def _on_run(self):
        try:
            argv = self.build_argv()
        except ValueError as e:
            messagebox.showerror("Fehlende Angaben", str(e))
            return

        try:
            namespace = self.parser.parse_args(argv)
        except SystemExit:
            # argparse hat selbst einen Fehler ausgegeben (z.B. invalid choice)
            messagebox.showerror("Ungültige Argumente", "Die eingegebenen Werte konnten nicht geparst werden.")
            return

        self._append_output(f"$ {self.parser.prog} {' '.join(shlex.quote(a) for a in argv)}\n")

        if self.run_callback is not None:
            try:
                #!                result = self.run_callback(namespace)
                result = self.run_callback(argv)
                if result is not None:
                    self._append_output(str(result) + "\n")
            except Exception as e:  # noqa: BLE001
                self._append_output(f"Fehler: {e}\n")
                messagebox.showerror("Fehler bei der Ausführung", str(e))
            return

        if self.script_path:
            self._run_subprocess([sys.executable, self.script_path] + argv)
            return

        # Fallback: nur anzeigen
        self._append_output(
            "(Kein run_callback und kein script_path gesetzt – " "nur die Argumente wurden ermittelt.)\n"
        )

    def _run_subprocess(self, cmd: list[str]):
        self._append_output(f"Starte: {' '.join(shlex.quote(c) for c in cmd)}\n")
        self.root.update_idletasks()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                self._append_output(proc.stdout)
            if proc.stderr:
                self._append_output(proc.stderr)
            self._append_output(f"[Beendet mit Code {proc.returncode}]\n")
        except Exception as e:  # noqa: BLE001
            self._append_output(f"Fehler beim Start des Subprozesses: {e}\n")

    def _append_output(self, text: str):
        if not self.capture_output:
            return
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")


# --------------------------------------------------------------------------
# Komfortfunktion
# --------------------------------------------------------------------------


def build_gui(
    parser: argparse.ArgumentParser,
    run_callback: Optional[Callable[[argparse.Namespace], Any]] = None,
    **kwargs,
) -> ArgparseGUI:
    """Kurzform: erstellt eine ArgparseGUI und startet sie noch nicht
    (damit man bei Bedarf vorher noch etwas konfigurieren kann)."""
    return ArgparseGUI(parser, run_callback=run_callback, **kwargs)


# --------------------------------------------------------------------------
# Demo / Selbsttest, wenn das Modul direkt ausgeführt wird
# --------------------------------------------------------------------------

if __name__ == "__main__":
    demo_parser = argparse.ArgumentParser(
        prog="demo-tool", description="Demo-Programm zur Veranschaulichung von argparse_tk_gui."
    )
    demo_parser.add_argument("input", help="Eingabedatei, die verarbeitet wird")
    demo_parser.add_argument("-o", "--output", default="out.txt", help="Pfad der Ausgabedatei")
    demo_parser.add_argument("-v", "--verbose", action="store_true", help="Ausführliche Ausgabe aktivieren")
    demo_parser.add_argument("--mode", choices=["fast", "slow", "auto"], default="auto", help="Verarbeitungsmodus")
    demo_parser.add_argument(
        "--tags", action="append", default=[], help="Tags (mehrfach angebbar, kommagetrennt eintippen)"
    )
    demo_parser.add_argument("--level", type=int, default=1, help="Detailgrad (0-10)")

    def demo_main(args: argparse.Namespace):
        print("Demo gestartet mit:", args)
        return f"OK: {args}"

    gui = ArgparseGUI(demo_parser, run_callback=demo_main, title="argparse_tk_gui – Demo")
    gui.mainloop()
