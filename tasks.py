from invoke import task

@task(default=True)
def main(ctx):
    ctx.run("python src")