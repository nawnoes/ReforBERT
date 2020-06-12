from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep


mb = master_bar(range(10))
for i in mb:
    for j in enumerate(progress_bar(range(100), parent=mb)):
        sleep(0.01)
        # mb.child.comment = f'second bar stat'
    # mb.main_bar.comment = f'first bar stat'
    # mb.write(f'Finished loop {i}.')
    #mb.update_graph(graphs, x_bounds, y_bounds)