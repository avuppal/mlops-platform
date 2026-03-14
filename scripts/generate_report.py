from src.registry import ModelRegistry
import os

def generate_comparison_report(registry_path="models.db", output_path="report.html"):
    registry = ModelRegistry(registry_path)
    models = registry.list_models()
    
    html = """
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h2>Model Comparison Report</h2>
        <table>
            <tr>
                <th>Model Name</th>
                <th>Version</th>
                <th>Stage</th>
                <th>Metrics</th>
            </tr>
    """
    
    for m in models:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in m.get('metrics', {}).items()])
        html += f"""
            <tr>
                <td>{m['name']}</td>
                <td>{m['version']}</td>
                <td>{m['stage']}</td>
                <td>{metrics_str}</td>
            </tr>
        """
        
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, "w") as f:
        f.write(html)
        
    print(f"Report generated at {output_path}")

if __name__ == "__main__":
    generate_comparison_report()
