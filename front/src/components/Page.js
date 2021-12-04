import Footer from "./Footer";
import Header from "./Header";


export default function Page({children}) {
    return (
        <div style={{padding:40}}>
            <Header></Header>
            <div>
                {children}
            </div>
            <Footer></Footer>
        </div>
    )
}