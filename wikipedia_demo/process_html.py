import os
from bs4 import BeautifulSoup

def process_html(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find the body tag
    body = soup.find('body')
    if not body:
        print("Error: No <body> tag found in the HTML.")
        return

    # Find all top-level details within the body
    top_level_details = body.find_all('details', recursive=False)

    print(f"Found {len(top_level_details)} root detail element(s)")
    
    for details in top_level_details:
        process_details(soup, details, output_dir, 0)

    # Create the skeleton HTML
    skeleton = create_skeleton(soup, top_level_details)
    
    with open(os.path.join(output_dir, 'skeleton.html'), 'w', encoding='utf-8') as f:
        f.write(str(skeleton))

def process_details(soup, details, output_dir, depth):
    if depth >= 4:
        # Save this subtree as a fragment
        fragment_id = f"fragment_{hash(details.prettify())}"
        details['open'] = ''
        with open(os.path.join(output_dir, f"{fragment_id}.html"), 'w', encoding='utf-8') as f:
            f.write(str(details))
        
        summary = details.find('summary')
        details.clear()
        if summary:
            details.append(summary)
        
        del details['open']
        
        loading_text = soup.new_tag('p')
        loading_text.string = "Loading..."
        details.append(loading_text)
        
        print(str(details))
        
        # replace the content with an htmx placeholder
        details['hx-trigger'] = 'click'
        details['hx-get'] = f"/fragment/{fragment_id}"
        details['hx-target'] = "this"
        details['hx-swap'] = "outerHTML"
    else:
        for child in details.find_all('details', recursive=False):
            process_details(soup, child, output_dir, depth + 1)

def create_skeleton(soup, top_level_details):
    skeleton = soup.new_tag('html')
    head = soup.new_tag('head')
    skeleton.append(head)

    # Add necessary scripts and styles
    script = soup.new_tag('script', src="https://unpkg.com/htmx.org@1.9.6")
    head.append(script)
    style = soup.find('style')
    if style:
        head.append(style)

    body = soup.new_tag('body')
    skeleton.append(body)
    for details in top_level_details:
        body.append(details)


    return skeleton

if __name__ == "__main__":
    process_html("output_1mil_named.html", "fragments")
